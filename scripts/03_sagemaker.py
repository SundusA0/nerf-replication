"""Run the training script as an Amazon SageMaker training job.

    python3 scripts/03_sagemaker.py launch benchmark
    python3 scripts/03_sagemaker.py status
    python3 scripts/03_sagemaker.py stop

launch   uploads the code, starts a job and follows its log until the job
         ends. Control-C stops following; the job itself keeps running.
status   shows the state of the most recent job and the end of its log.
stop     asks SageMaker to stop the most recent job. The training script
         finishes its current step and saves a checkpoint first.

Needs the AWS SDK for Python (boto3) and AWS credentials. AWS CloudShell, the
terminal inside the AWS console, has both. PyTorch is not needed here: this
script only describes the job, and `nerf.sagemaker` explains how.

One-time setup, all in region us-east-1 (the README has the commands):

  * an S3 bucket named sagemaker-us-east-1-<account id>, holding the scene
    under nerf/data/lego/
  * an IAM role named nerf-sagemaker-role that SageMaker may assume, with the
    AmazonSageMakerFullAccess policy attached
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from nerf import sagemaker as job_spec
except ImportError:  # a plain checkout without `pip install -e .`, e.g. in CloudShell
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from nerf import sagemaker as job_spec

warnings.filterwarnings("ignore", message=".*Boto3 will no longer support Python.*")
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, ParamValidationError
except ImportError:
    boto3 = None

ROLE_NAME = "nerf-sagemaker-role"
ROLE_POLICY = "AmazonSageMakerFullAccess"
LOG_GROUP = "/aws/sagemaker/TrainingJobs"
FINISHED = ("Completed", "Failed", "Stopped")
# How this script was called, for the commands it suggests in its messages.
THIS_SCRIPT = "python3 " + (sys.argv[0] if sys.argv[0].endswith("03_sagemaker.py") else "scripts/03_sagemaker.py")


def make_clients() -> dict:
    session = boto3.session.Session(region_name=job_spec.REGION)
    return {name: session.client(name) for name in ("sts", "s3", "iam", "sagemaker", "logs")}


def error_text(error) -> str:
    details = error.response.get("Error", {})
    return f"{details.get('Code', 'Error')}: {details.get('Message', error)}"


def git_commit(repo_root: Path) -> str:
    """Short hash of the checked-out commit, marked if the code differs from it."""
    def git(*command: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *command], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "--short=12", "HEAD")
        changed = git("status", "--porcelain", "--", "src", "scripts")  # includes new files
        return commit + ("-modified" if changed else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def s3_has(s3, bucket: str, key: str) -> bool:
    listing = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return any(item["Key"] == key for item in listing.get("Contents", []))


def jobs_of_run(sagemaker, run: str, status: str) -> list:
    """Names of this run's jobs that are in the given state."""
    pattern = re.compile(rf"nerf-{re.escape(run)}-\d{{8}}-\d{{6}}")
    listing = sagemaker.list_training_jobs(
        NameContains=f"nerf-{run}-", StatusEquals=status, MaxResults=100
    )
    names = [summary["TrainingJobName"] for summary in listing["TrainingJobSummaries"]]
    return [name for name in names if pattern.fullmatch(name)]


def preflight(aws: dict, bucket: str, scene: str, run: str, out=print) -> list:
    """Check what a job depends on before anything is uploaded or started.

    Returns a list of problems; empty means the job can be launched.
    """
    problems = []

    scene_key = f"nerf/data/{scene}/transforms_train.json"
    try:
        if s3_has(aws["s3"], bucket, scene_key):
            out(f"  ok    scene found at s3://{bucket}/nerf/data/{scene}/")
        else:
            problems.append(f"no scene at s3://{bucket}/nerf/data/{scene}/ (looked for {scene_key})")
    except ClientError as error:
        problems.append(f"cannot read bucket {bucket}: {error_text(error)}")

    try:
        aws["iam"].get_role(RoleName=ROLE_NAME)
        attached = aws["iam"].list_attached_role_policies(RoleName=ROLE_NAME)["AttachedPolicies"]
        if any(policy["PolicyName"] == ROLE_POLICY for policy in attached):
            out(f"  ok    role {ROLE_NAME} exists and has {ROLE_POLICY}")
        else:
            problems.append(f"role {ROLE_NAME} does not have the {ROLE_POLICY} policy attached")
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") == "NoSuchEntity":
            problems.append(f"role {ROLE_NAME} does not exist")
        else:  # not allowed to look: let SageMaker decide when the job is created
            out(f"  note  could not inspect role {ROLE_NAME}: {error_text(error)}")

    try:
        running = jobs_of_run(aws["sagemaker"], run, "InProgress")
        running += jobs_of_run(aws["sagemaker"], run, "Stopping")
        if running:
            problems.append(
                f"job {running[0]} of run {run!r} is still active. Two jobs must not write "
                f"to one run directory. Wait for it, or stop it with: {THIS_SCRIPT} stop"
            )
        else:
            out(f"  ok    no other job of run {run!r} is active")
    except ClientError as error:
        problems.append(f"cannot list training jobs: {error_text(error)}")

    return problems


def upload_code(s3, bucket: str, key_prefix: str, files: dict) -> None:
    for relative_path, local_path in files.items():
        s3.put_object(Bucket=bucket, Key=key_prefix + relative_path, Body=local_path.read_bytes())


def split_s3_uri(uri: str):
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


# ----------------------------------------------------------------------------
# Following a job
# ----------------------------------------------------------------------------


def log_stream_of(logs, job_name: str):
    """Name of the job's log stream, or None if it has not started logging."""
    try:
        streams = logs.describe_log_streams(
            logGroupName=LOG_GROUP, logStreamNamePrefix=job_name + "/"
        )["logStreams"]
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            return None  # the log group itself appears with the account's first job
        raise
    return streams[0]["logStreamName"] if streams else None


def new_log_lines(logs, stream: str, token):
    """Log lines after `token` (None: from the start). Returns (lines, new token)."""
    lines = []
    for _ in range(100):  # pages per call; the rest is picked up on the next call
        request = {"logGroupName": LOG_GROUP, "logStreamName": stream, "startFromHead": True}
        if token:
            request["nextToken"] = token
        page = logs.get_log_events(**request)
        lines += [event["message"].rstrip() for event in page["events"]]
        if page["nextForwardToken"] == token:  # the API's signal for "nothing newer"
            break
        token = page["nextForwardToken"]
    return lines, token


def last_log_lines(logs, job_name: str, count: int) -> list:
    stream = log_stream_of(logs, job_name)
    if stream is None:
        return []
    page = logs.get_log_events(
        logGroupName=LOG_GROUP, logStreamName=stream, limit=count, startFromHead=False
    )
    return [event["message"].rstrip() for event in page["events"]]


def print_summary(description: dict, out=print) -> None:
    status = description["TrainingJobStatus"]
    out(f"job      {description['TrainingJobName']}")
    out(f"status   {status} ({description.get('SecondaryStatus', '')})")
    out(f"machine  {description['ResourceConfig']['InstanceType']}, time limit "
        f"{description['StoppingCondition']['MaxRuntimeInSeconds'] / 60:.0f} min")
    if "BillableTimeInSeconds" in description:
        out(f"billed   {description['BillableTimeInSeconds'] / 60:.1f} min")
    elif "TrainingStartTime" in description and status not in FINISHED:
        running = datetime.now(timezone.utc) - description["TrainingStartTime"]
        out(f"running  {running.total_seconds() / 60:.1f} min so far")
    if description.get("FailureReason"):
        out(f"reason   {description['FailureReason']}")
    if "CheckpointConfig" in description:
        out(f"results  {description['CheckpointConfig']['S3Uri']}")


def watch(aws: dict, job_name: str, out=print, sleep=time.sleep, poll_seconds: float = 15) -> int:
    """Print the job's progress and log until it ends. Returns 0 if it completed."""
    out("Following the job. Control-C stops following; the job itself keeps running.")
    shown, stream, token = set(), None, None

    def show_new_log_lines():
        nonlocal stream, token
        stream = stream or log_stream_of(aws["logs"], job_name)
        if stream:
            lines, token = new_log_lines(aws["logs"], stream, token)
            for line in lines:
                out(line)

    try:
        while True:
            description = aws["sagemaker"].describe_training_job(TrainingJobName=job_name)
            for step in description.get("SecondaryStatusTransitions", []):
                key = (step["Status"], step.get("StatusMessage", ""))
                if key not in shown:
                    shown.add(key)
                    out(f"[{step['StartTime'].astimezone(timezone.utc):%H:%M:%S} UTC] "
                        f"{step['Status']}: {step.get('StatusMessage', '')}")
            show_new_log_lines()
            if description["TrainingJobStatus"] in FINISHED:
                break
            sleep(poll_seconds)
        sleep(min(poll_seconds, 5))  # the last log lines arrive a little after the job ends
        show_new_log_lines()
    except KeyboardInterrupt:
        out(f"\nStopped following. The job is still running; check it with: {THIS_SCRIPT} status")
        return 0

    out("")
    print_summary(description, out)
    return 0 if description["TrainingJobStatus"] == "Completed" else 1


def latest_job(sagemaker, status=None):
    """Name of the most recently created job of this project, or None."""
    request = {"NameContains": "nerf-", "SortBy": "CreationTime", "SortOrder": "Descending",
               "MaxResults": 100}
    if status:
        request["StatusEquals"] = status
    jobs = sagemaker.list_training_jobs(**request)["TrainingJobSummaries"]
    jobs = [job for job in jobs if job["TrainingJobName"].startswith("nerf-")]
    if not jobs:
        return None
    return max(jobs, key=lambda job: job["CreationTime"])["TrainingJobName"]


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------


def launch(args, aws: dict, out=print, sleep=time.sleep) -> int:
    preset = job_spec.PRESETS[args.preset]
    iterations = args.iterations if args.iterations is not None else preset.iterations
    if args.max_hours is not None:
        limit_minutes = args.max_hours * 60
    elif preset.max_runtime_minutes is not None:
        limit_minutes = preset.max_runtime_minutes
    else:
        out(f"The {args.preset!r} preset has no default time limit. Choose one with --max-hours; "
            "the job is stopped, with a checkpoint, when it is reached.")
        return 2

    account = aws["sts"].get_caller_identity()["Account"]
    bucket = args.bucket or f"sagemaker-{job_spec.REGION}-{account}"
    job_name = job_spec.make_job_name(args.run, datetime.now(timezone.utc))
    device = "cuda" if job_spec.is_gpu_instance(args.instance_type) else "cpu"
    commit = git_commit(REPO_ROOT)
    request = job_spec.training_job_request(
        job_name=job_name, run=args.run, bucket=bucket,
        role_arn=f"arn:aws:iam::{account}:role/{ROLE_NAME}",
        instance_type=args.instance_type,
        max_runtime_seconds=round(limit_minutes * 60),
        arguments=job_spec.training_arguments(
            iterations, preset.validate_every, preset.checkpoint_every, device,
            extra=tuple(shlex.split(args.extra)),
        ),
        scene=args.scene,
        tags={"project": "nerf-replication", "run": args.run, "git-commit": commit},
    )
    if args.dry_run:
        out(json.dumps(request, indent=2))
        return 0

    out(f"Checking the setup in account {account}, region {job_spec.REGION}:")
    problems = preflight(aws, bucket, args.scene, args.run, out)
    if problems:
        for problem in problems:
            out(f"  FAIL  {problem}")
        out("Nothing was started.")
        return 1

    where = job_spec.s3_layout(bucket, args.run, job_name, args.scene)
    _, run_key = split_s3_uri(where["run"])
    if s3_has(aws["s3"], bucket, f"{run_key}/checkpoint.pt"):
        out(f"  note  run {args.run!r} already has a checkpoint; this job continues it")
    else:
        out(f"  note  run {args.run!r} has no checkpoint yet; this job starts it")

    files = job_spec.code_bundle(REPO_ROOT)
    _, code_key = split_s3_uri(where["code"])
    upload_code(aws["s3"], bucket, code_key, files)
    out(f"Uploaded {len(files)} code files (commit {commit}) to {where['code']}")

    try:
        aws["sagemaker"].create_training_job(**request)
    except ParamValidationError as error:
        out(f"This version of boto3 rejected the request before sending it:\n{error}\n"
            "Nothing was started. Updating boto3 may help: pip3 install --user --upgrade boto3")
        return 1
    except ClientError as error:
        out(f"AWS refused to start the job. {error_text(error)}")
        out("Nothing was started, so nothing is being charged.")
        return 1

    out(f"Started job {job_name}")
    out(f"  machine     {args.instance_type}, stopped automatically after {limit_minutes:.0f} min")
    out(f"  program     python3 -u {job_spec.ENTRY_SCRIPT} " + " ".join(request["AlgorithmSpecification"]["ContainerArguments"]))
    out(f"  results     {where['run']}")
    out(f"  to stop it  {THIS_SCRIPT} stop")
    if args.no_watch:
        return 0
    return watch(aws, job_name, out, sleep)


def status(args, aws: dict, out=print, sleep=time.sleep) -> int:
    job_name = args.job or latest_job(aws["sagemaker"])
    if job_name is None:
        out("No training job of this project exists yet.")
        return 1
    if args.watch:
        return watch(aws, job_name, out, sleep)
    description = aws["sagemaker"].describe_training_job(TrainingJobName=job_name)
    print_summary(description, out)
    lines = last_log_lines(aws["logs"], job_name, args.lines)
    out("")
    out(f"last {len(lines)} log lines:" if lines else "no log lines yet")
    for line in lines:
        out("  " + line)
    return 0


def stop(args, aws: dict, out=print) -> int:
    job_name = args.job or latest_job(aws["sagemaker"], status="InProgress")
    if job_name is None:
        out("No training job of this project is running.")
        return 0
    try:
        aws["sagemaker"].stop_training_job(TrainingJobName=job_name)
    except ClientError as error:
        out(f"Could not stop {job_name}. {error_text(error)}")
        return 1
    out(f"Asked SageMaker to stop {job_name}. It saves a checkpoint and stops within a few "
        f"minutes; confirm with: {THIS_SCRIPT} status")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("launch", help="start a training job")
    p.add_argument("preset", choices=sorted(job_spec.PRESETS))
    p.add_argument("--run", default="lego-800px",
                   help="name of the training run; jobs with the same name continue one another")
    p.add_argument("--scene", default="lego", help="scene directory under nerf/data/ in the bucket")
    p.add_argument("--iterations", type=int, help="total training steps (default: the preset's)")
    p.add_argument("--max-hours", type=float, help="stop the job after this long")
    p.add_argument("--instance-type", default="ml.g5.xlarge")
    p.add_argument("--extra", default="", metavar="ARGS",
                   help='more arguments for 02_train.py, quoted, e.g. --extra="--downscale 2". '
                        "Settings that change the model or the sampling need a new --run name")
    p.add_argument("--bucket", help="default: sagemaker-us-east-1-<account id>")
    p.add_argument("--no-watch", action="store_true", help="return as soon as the job is started")
    p.add_argument("--dry-run", action="store_true",
                   help="print the request; upload nothing and start nothing")

    p = commands.add_parser("status", help="state and recent log of a job")
    p.add_argument("job", nargs="?", help="job name (default: the most recent)")
    p.add_argument("--watch", action="store_true", help="keep following until the job ends")
    p.add_argument("--lines", type=int, default=15, help="log lines to show")

    p = commands.add_parser("stop", help="stop a running job")
    p.add_argument("job", nargs="?", help="job name (default: the most recent running one)")

    args = parser.parse_args(argv)
    if boto3 is None:
        print("This script needs the AWS SDK for Python: pip3 install boto3")
        return 2
    try:
        aws = make_clients()
        return {"launch": launch, "status": status, "stop": stop}[args.command](args, aws)
    except ValueError as error:
        print(error)
        return 2
    except ClientError as error:
        print(f"AWS error. {error_text(error)}")
        return 1
    except BotoCoreError as error:  # e.g. no credentials, no network
        print(f"Could not reach AWS: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
