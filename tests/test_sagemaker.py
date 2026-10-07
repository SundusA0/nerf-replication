import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ParamValidationError
from botocore.stub import Stubber
from botocore.validate import validate_parameters

from nerf import sagemaker as job_spec
from nerf.data import analytic_views
from test_data import write_blender_scene

ROOT = Path(__file__).resolve().parents[1]
WHEN = datetime(2026, 10, 7, 17, 25, 1, tzinfo=timezone.utc)
ACCOUNT = "123456789012"
BUCKET = f"sagemaker-us-east-1-{ACCOUNT}"


def make_request(**overrides):
    settings = dict(
        job_name=job_spec.make_job_name("lego-800px", WHEN),
        run="lego-800px",
        bucket=BUCKET,
        role_arn=f"arn:aws:iam::{ACCOUNT}:role/nerf-sagemaker-role",
        instance_type="ml.g5.xlarge",
        max_runtime_seconds=45 * 60,
        arguments=job_spec.training_arguments(2000, 1000, 500, "cuda"),
        tags={"project": "nerf-replication", "git-commit": "0123456789ab"},
    )
    settings.update(overrides)
    return job_spec.training_job_request(**settings)


# --------------------------------------------------------------------------
# Describing a job (no AWS involved)
# --------------------------------------------------------------------------


def test_job_names():
    assert job_spec.make_job_name("lego-800px", WHEN) == "nerf-lego-800px-20261007-172501"
    longest = "a" * 42
    assert len(job_spec.make_job_name(longest, WHEN)) == 63
    for bad in ("", "Lego", "lego_800px", "lego-", "-lego", "lego--800px", "lego 800px", "a" * 43):
        with pytest.raises(ValueError):
            job_spec.make_job_name(bad, WHEN)


def test_image_matches_the_machine():
    assert job_spec.is_gpu_instance("ml.g5.xlarge") and job_spec.is_gpu_instance("ml.p3.2xlarge")
    assert not job_spec.is_gpu_instance("ml.m5.xlarge") and not job_spec.is_gpu_instance("ml.c5.4xlarge")
    gpu, cpu = job_spec.image_uri("ml.g5.xlarge"), job_spec.image_uri("ml.m5.xlarge")
    assert gpu.startswith("763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:")
    assert "-gpu-" in gpu and "-cpu-" in cpu
    assert gpu.endswith("-sagemaker") and cpu.endswith("-sagemaker")
    for bad in ("g5.xlarge", "ml.g5", "ml..xlarge", ""):
        with pytest.raises(ValueError):
            job_spec.is_gpu_instance(bad)


def test_request_describes_the_same_program_as_a_local_run():
    request = make_request()
    spec = request["AlgorithmSpecification"]
    assert spec["ContainerEntrypoint"] == ["python3", "-u", "/opt/ml/input/data/code/02_train.py"]
    assert spec["ContainerArguments"] == [
        "/opt/ml/input/data/training",
        "--out", "/opt/ml/checkpoints",
        "--iterations", "2000",
        "--validate-every", "1000",
        "--checkpoint-every", "500",
        "--device", "cuda",
    ]
    assert spec["TrainingImage"] == job_spec.image_uri("ml.g5.xlarge")
    assert request["ResourceConfig"]["InstanceType"] == "ml.g5.xlarge"
    assert request["ResourceConfig"]["InstanceCount"] == 1
    assert request["StoppingCondition"] == {"MaxRuntimeInSeconds": 2700}
    assert request["RoleArn"].endswith(":role/nerf-sagemaker-role")
    assert request["Tags"] == [
        {"Key": "git-commit", "Value": "0123456789ab"},
        {"Key": "project", "Value": "nerf-replication"},
    ]


def test_every_flag_exists_in_the_training_script():
    script = (ROOT / "scripts" / "02_train.py").read_text()
    flags = [a for a in job_spec.training_arguments(1, 1, 1, "cpu") if a.startswith("--")]
    assert flags == ["--out", "--iterations", "--validate-every", "--checkpoint-every", "--device"]
    for flag in flags:
        assert f'"{flag}"' in script, flag


def test_channels_and_run_directory():
    request = make_request()
    channels = {c["ChannelName"]: c for c in request["InputDataConfig"]}
    assert set(channels) == {"training", "code"}
    for channel in channels.values():
        assert channel["InputMode"] == "File"
        assert channel["DataSource"]["S3DataSource"]["S3DataType"] == "S3Prefix"
    uri = lambda name: channels[name]["DataSource"]["S3DataSource"]["S3Uri"]
    assert uri("training") == f"s3://{BUCKET}/nerf/data/lego/"
    assert uri("code") == f"s3://{BUCKET}/nerf/code/nerf-lego-800px-20261007-172501/"
    # the channel names are the last part of the paths the program is given
    assert job_spec.DATA_DIR == "/opt/ml/input/data/training"
    assert job_spec.CODE_DIR == "/opt/ml/input/data/code"

    # The script's --out directory is the directory SageMaker copies to S3.
    arguments = request["AlgorithmSpecification"]["ContainerArguments"]
    out_dir = arguments[arguments.index("--out") + 1]
    assert request["CheckpointConfig"] == {
        "S3Uri": f"s3://{BUCKET}/nerf/runs/lego-800px/out",
        "LocalPath": out_dir,
    }


def test_jobs_of_one_run_share_a_run_directory_but_not_code():
    def request_for(run, hour=17):
        return make_request(run=run, job_name=job_spec.make_job_name(run, WHEN.replace(hour=hour)))

    run_dir = lambda r: r["CheckpointConfig"]["S3Uri"]
    code = lambda r: r["InputDataConfig"][1]["DataSource"]["S3DataSource"]["S3Uri"]
    first, later = request_for("lego-800px"), request_for("lego-800px", hour=23)
    assert run_dir(first) == run_dir(later)       # a later job continues the run ...
    assert code(first) != code(later)             # ... with its own copy of the code

    # Different runs never share files, even when one name starts with the
    # other: no run directory is a string prefix of another.
    directories = [run_dir(request_for(run)) for run in ("lego", "lego-800px", "lego-800px-b")]
    for a in directories:
        for b in directories:
            assert a == b or not b.startswith(a)


def test_presets():
    benchmark, full = job_spec.PRESETS["benchmark"], job_spec.PRESETS["full"]
    assert full.iterations == 500_000                 # the released paper configuration
    assert benchmark.iterations < full.iterations
    assert benchmark.max_runtime_minutes is not None  # a cheap job has a default cap
    assert full.max_runtime_minutes is None           # an expensive one must be given its cap
    for preset in (benchmark, full):
        assert preset.iterations % preset.checkpoint_every == 0


def test_limits_are_enforced():
    with pytest.raises(ValueError, match="256"):
        make_request(arguments=["x" * 257])
    with pytest.raises(ValueError, match="items"):
        make_request(arguments=["x"] * 101)
    with pytest.raises(ValueError, match="items"):
        make_request(arguments=[])
    with pytest.raises(ValueError, match="name"):
        make_request(job_name="nerf_bad_name")
    with pytest.raises(ValueError, match="time limit"):
        make_request(max_runtime_seconds=0)


def test_code_bundle_is_the_script_next_to_the_package():
    files = job_spec.code_bundle(ROOT)
    assert files["02_train.py"] == ROOT / "scripts" / "02_train.py"
    package = sorted(name for name in files if name.startswith("nerf/"))
    assert package == sorted(f"nerf/{p.name}" for p in (ROOT / "src" / "nerf").glob("*.py"))
    assert {"nerf/__init__.py", "nerf/train.py", "nerf/render.py", "nerf/data.py"} <= set(files)
    assert all(name.endswith(".py") for name in files)
    with pytest.raises(FileNotFoundError):
        job_spec.code_bundle(ROOT / "tests")


def test_the_module_needs_neither_torch_nor_the_aws_sdk():
    # The launcher runs where PyTorch is not installed (AWS CloudShell).
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['boto3'] = None; "
        "sys.modules['numpy'] = None; from nerf import sagemaker; print('ok')"
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.stdout.strip() == "ok", result.stderr


# --------------------------------------------------------------------------
# The job's command line, run for real on a copy of the uploaded files
# --------------------------------------------------------------------------


def test_the_job_command_runs_on_the_uploaded_files(tmp_path):
    # Lay out what the container would see: the code bundle in one directory,
    # a scene in another, an empty run directory.
    code_dir, data_dir, run_dir = tmp_path / "code", tmp_path / "training", tmp_path / "checkpoints"
    for relative, source in job_spec.code_bundle(ROOT).items():
        target = code_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
    # Leave a trace when the copied package is imported, to prove the job
    # would use the uploaded code and not an installed copy.
    marker = tmp_path / "imported_from.txt"
    with (code_dir / "nerf" / "__init__.py").open("a") as handle:
        handle.write(f"\nopen({str(marker)!r}, 'w').write(__file__)\n")
    for split in ("train", "val"):
        write_blender_scene(data_dir, analytic_views(4, image_size=16, num_samples=32), split)

    tiny = ("--depth", "2", "--width", "16", "--skip-layer", "0", "--num-coarse", "4",
            "--num-fine", "4", "--batch-size", "32", "--val-skip", "2")
    request = make_request(
        instance_type="ml.m5.xlarge",
        arguments=job_spec.training_arguments(4, 2, 2, "cpu", extra=tiny),
    )
    spec = request["AlgorithmSpecification"]
    command = spec["ContainerEntrypoint"] + spec["ContainerArguments"]
    assert command[0] == "python3"
    local = {job_spec.CODE_DIR: code_dir, job_spec.DATA_DIR: data_dir, job_spec.RUN_DIR: run_dir}
    for container_path, local_path in local.items():
        command = [part.replace(container_path, str(local_path)) for part in command]
    command[0] = sys.executable

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "finished at step 4" in result.stdout
    assert Path(marker.read_text()).parent.resolve() == (code_dir / "nerf").resolve()
    produced = {p.name for p in run_dir.iterdir()}
    assert {"checkpoint.pt", "config.json", "log.csv", "validation.csv", "summary.json",
            "val_000002.png", "val_000004.png"} <= produced

    # Launching the job again with a larger step count continues the run.
    command[command.index("--iterations") + 1] = "6"
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "resumed from" in result.stdout and "at step 4" in result.stdout
    assert "finished at step 6" in result.stdout


# --------------------------------------------------------------------------
# The request against the AWS SDK's own description of the API
# --------------------------------------------------------------------------


def clients():
    session = boto3.session.Session(
        region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test"
    )
    return {name: session.client(name) for name in ("sts", "s3", "iam", "sagemaker", "logs")}


def violations(value, shape, path="request"):
    """Length, range, pattern and enum rules of the API model that `value` breaks.

    The SDK itself only checks types and required fields before sending.
    """
    found, checked = [], 0
    rules = shape.metadata
    if shape.type_name == "structure":
        for key, member in value.items():
            f, c = violations(member, shape.members[key], f"{path}.{key}")
            found += f
            checked += c
    elif shape.type_name == "list":
        for index, item in enumerate(value):
            f, c = violations(item, shape.member, f"{path}[{index}]")
            found += f
            checked += c
    elif shape.type_name == "map":
        for key, item in value.items():
            f1, c1 = violations(key, shape.key, f"{path} key {key!r}")
            f2, c2 = violations(item, shape.value, f"{path}[{key!r}]")
            found += f1 + f2
            checked += c1 + c2
    if shape.type_name in ("string", "list", "map"):
        size = len(value)
        if "min" in rules and size < rules["min"]:
            found.append(f"{path}: length {size} < {rules['min']}")
        if "max" in rules and size > rules["max"]:
            found.append(f"{path}: length {size} > {rules['max']}")
        checked += ("min" in rules) + ("max" in rules)
    if shape.type_name in ("integer", "long"):
        if "min" in rules and value < rules["min"]:
            found.append(f"{path}: {value} < {rules['min']}")
        if "max" in rules and value > rules["max"]:
            found.append(f"{path}: {value} > {rules['max']}")
        checked += ("min" in rules) + ("max" in rules)
    if shape.type_name == "string":
        if "enum" in rules:
            checked += 1
            if value not in rules["enum"]:
                found.append(f"{path}: {value!r} is not one of the allowed values")
        if "pattern" in rules:
            try:
                pattern = re.compile(rules["pattern"])
            except re.error:  # a few patterns use syntax Python's re does not have
                pattern = None
            if pattern is not None:
                checked += 1
                if not pattern.fullmatch(value):
                    found.append(f"{path}: {value!r} does not match {rules['pattern']}")
    return found, checked


@pytest.mark.parametrize("instance_type", ["ml.g5.xlarge", "ml.m5.xlarge"])
def test_request_satisfies_the_api_model(instance_type):
    request = make_request(instance_type=instance_type)
    shape = clients()["sagemaker"].meta.service_model.operation_model("CreateTrainingJob").input_shape

    validate_parameters(request, shape)   # raises on wrong types, unknown or missing fields
    found, checked = violations(request, shape)
    assert found == []
    assert checked > 40                             # the walk really did visit the rules


def test_the_checks_catch_violations():
    shape = clients()["sagemaker"].meta.service_model.operation_model("CreateTrainingJob").input_shape
    with pytest.raises(ParamValidationError):
        validate_parameters({**make_request(), "NoSuchField": 1}, shape)
    with pytest.raises(ParamValidationError):
        validate_parameters({k: v for k, v in make_request().items() if k != "RoleArn"}, shape)

    request = make_request()
    request["TrainingJobName"] = "x" * 64
    request["AlgorithmSpecification"]["ContainerArguments"] = ["y" * 257]
    request["AlgorithmSpecification"]["TrainingInputMode"] = "Disk"
    request["InputDataConfig"][0]["ChannelName"] = "has space"
    request["StoppingCondition"]["MaxRuntimeInSeconds"] = 0
    found, _ = violations(request, shape)
    assert len(found) >= 5


# --------------------------------------------------------------------------
# The launcher script, with AWS replaced by canned responses
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("sagemaker_cli", ROOT / "scripts" / "03_sagemaker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def description(job_name, status, secondary, transitions=(), **extra):
    request = make_request(job_name=job_name)
    return {
        "TrainingJobName": job_name,
        "TrainingJobArn": f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:training-job/{job_name}",
        "ModelArtifacts": {"S3ModelArtifacts": f"s3://{BUCKET}/nerf/output/{job_name}/output/model.tar.gz"},
        "TrainingJobStatus": status,
        "SecondaryStatus": secondary,
        "AlgorithmSpecification": request["AlgorithmSpecification"],
        "ResourceConfig": request["ResourceConfig"],
        "StoppingCondition": request["StoppingCondition"],
        "CheckpointConfig": request["CheckpointConfig"],
        "CreationTime": WHEN,
        "SecondaryStatusTransitions": [
            {"Status": s, "StartTime": WHEN, "StatusMessage": m} for s, m in transitions
        ],
        **extra,
    }


def log_page(messages, token):
    events = [{"timestamp": 1, "message": m, "ingestionTime": 1} for m in messages]
    return {"events": events, "nextForwardToken": token, "nextBackwardToken": "b/0"}


def test_watch_follows_a_job_to_its_end(cli):
    aws = clients()
    job = "nerf-lego-800px-20261007-172501"
    stream = {"logStreams": [{"logStreamName": f"{job}/algo-1-1700000000"}]}
    printed, naps = [], []
    # Each request for log lines must ask for what follows the last one seen.
    from_start = {"logGroupName": cli.LOG_GROUP, "logStreamName": f"{job}/algo-1-1700000000",
                  "startFromHead": True}
    after = lambda token: {**from_start, "nextToken": token}

    with Stubber(aws["sagemaker"]) as sagemaker, Stubber(aws["logs"]) as logs:
        # poll 1: machine being prepared, nothing logged yet
        sagemaker.add_response("describe_training_job", description(
            job, "InProgress", "Starting", [("Starting", "Preparing the instances for training")]))
        logs.add_response("describe_log_streams", {"logStreams": []})
        # poll 2: training, first log lines
        sagemaker.add_response("describe_training_job", description(
            job, "InProgress", "Training",
            [("Starting", "Preparing the instances for training"), ("Training", "Training in progress")]))
        logs.add_response("describe_log_streams", stream)
        logs.add_response("get_log_events", log_page(["step 100  loss 0.05\n", "step 200  loss 0.03"], "f/2"),
                          from_start)
        logs.add_response("get_log_events", log_page([], "f/2"), after("f/2"))
        # poll 3: finished; one more line had arrived, and one arrives after the end
        sagemaker.add_response("describe_training_job", description(
            job, "Completed", "Completed",
            [("Starting", "Preparing the instances for training"), ("Training", "Training in progress"),
             ("Completed", "Training job completed")], BillableTimeInSeconds=780))
        logs.add_response("get_log_events", log_page(["step 300  loss 0.02"], "f/3"), after("f/2"))
        logs.add_response("get_log_events", log_page([], "f/3"), after("f/3"))
        logs.add_response("get_log_events", log_page(["finished at step 300"], "f/4"), after("f/3"))
        logs.add_response("get_log_events", log_page([], "f/4"), after("f/4"))

        code = cli.watch(aws, job, out=printed.append, sleep=naps.append, poll_seconds=15)
        sagemaker.assert_no_pending_responses()
        logs.assert_no_pending_responses()

    assert code == 0
    text = "\n".join(printed)
    log_lines = [line for line in printed if line.startswith(("step", "finished"))]
    assert log_lines == ["step 100  loss 0.05", "step 200  loss 0.03", "step 300  loss 0.02",
                         "finished at step 300"]                      # each line once, in order
    for status in ("Starting: Preparing", "Training: Training in progress", "Completed: Training job"):
        assert text.count(status) == 1                               # each stage announced once
    assert "billed   13.0 min" in text
    assert naps == [15, 15, 5]                                       # waits between polls, then for the log tail


def test_watch_reports_failure(cli):
    aws = clients()
    job = "nerf-lego-800px-20261007-172501"
    printed = []
    with Stubber(aws["sagemaker"]) as sagemaker, Stubber(aws["logs"]) as logs:
        sagemaker.add_response("describe_training_job", description(
            job, "Failed", "Failed", [("Failed", "Training job failed")],
            FailureReason="AlgorithmError: , exit code: 1"))
        logs.add_client_error("describe_log_streams", "ResourceNotFoundException")
        logs.add_client_error("describe_log_streams", "ResourceNotFoundException")
        code = cli.watch(aws, job, out=printed.append, sleep=lambda seconds: None)
    assert code == 1
    assert "reason   AlgorithmError: , exit code: 1" in printed


def summaries(*names):
    return {"TrainingJobSummaries": [
        {"TrainingJobName": n, "TrainingJobArn": f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:training-job/{n}",
         "CreationTime": WHEN, "TrainingJobStatus": "InProgress"} for n in names]}


def test_preflight_passes_on_a_complete_setup(cli):
    aws = clients()
    printed = []
    with Stubber(aws["s3"]) as s3, Stubber(aws["iam"]) as iam, Stubber(aws["sagemaker"]) as sagemaker:
        s3.add_response("list_objects_v2", {"Contents": [{"Key": "nerf/data/lego/transforms_train.json"}]},
                        {"Bucket": BUCKET, "Prefix": "nerf/data/lego/transforms_train.json", "MaxKeys": 1})
        iam.add_response("get_role", {"Role": {
            "Path": "/", "RoleName": cli.ROLE_NAME, "RoleId": "AROAEXAMPLEEXAMPLE1",
            "Arn": f"arn:aws:iam::{ACCOUNT}:role/{cli.ROLE_NAME}", "CreateDate": WHEN}})
        iam.add_response("list_attached_role_policies", {"AttachedPolicies": [
            {"PolicyName": "AmazonSageMakerFullAccess",
             "PolicyArn": "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"}]})
        # A job of another run whose name merely contains this run's name is not a conflict.
        sagemaker.add_response("list_training_jobs", summaries("nerf-lego-800px-b-20261007-100000"))
        sagemaker.add_response("list_training_jobs", summaries())
        problems = cli.preflight(aws, BUCKET, "lego", "lego-800px", out=printed.append)
    assert problems == []
    assert len(printed) == 3 and all(line.startswith("  ok") for line in printed)


def test_preflight_names_every_problem(cli):
    aws = clients()
    with Stubber(aws["s3"]) as s3, Stubber(aws["iam"]) as iam, Stubber(aws["sagemaker"]) as sagemaker:
        s3.add_response("list_objects_v2", {})                              # no scene
        iam.add_response("get_role", {"Role": {
            "Path": "/", "RoleName": cli.ROLE_NAME, "RoleId": "AROAEXAMPLEEXAMPLE1",
            "Arn": f"arn:aws:iam::{ACCOUNT}:role/{cli.ROLE_NAME}", "CreateDate": WHEN}})
        iam.add_response("list_attached_role_policies", {"AttachedPolicies": []})   # no policy
        sagemaker.add_response("list_training_jobs", summaries("nerf-lego-800px-20261007-100000"))
        sagemaker.add_response("list_training_jobs", summaries())
        problems = cli.preflight(aws, BUCKET, "lego", "lego-800px", out=lambda line: None)
    assert len(problems) == 3
    assert "no scene" in problems[0]
    assert "AmazonSageMakerFullAccess" in problems[1]
    assert "nerf-lego-800px-20261007-100000" in problems[2] and "still active" in problems[2]


def test_preflight_reports_a_missing_role(cli):
    aws = clients()
    with Stubber(aws["s3"]) as s3, Stubber(aws["iam"]) as iam, Stubber(aws["sagemaker"]) as sagemaker:
        s3.add_response("list_objects_v2", {"Contents": [{"Key": "nerf/data/lego/transforms_train.json"}]})
        iam.add_client_error("get_role", "NoSuchEntity")
        sagemaker.add_response("list_training_jobs", summaries())
        sagemaker.add_response("list_training_jobs", summaries())
        problems = cli.preflight(aws, BUCKET, "lego", "lego-800px", out=lambda line: None)
    assert problems == [f"role {cli.ROLE_NAME} does not exist"]


def test_upload_puts_each_file_where_the_job_expects_it(cli):
    aws = clients()
    files = job_spec.code_bundle(ROOT)
    prefix = "nerf/code/nerf-lego-800px-20261007-172501/"
    with Stubber(aws["s3"]) as s3:
        for relative, path in files.items():
            s3.add_response("put_object", {}, {
                "Bucket": BUCKET, "Key": prefix + relative, "Body": path.read_bytes()})
        cli.upload_code(aws["s3"], BUCKET, prefix, files)
        s3.assert_no_pending_responses()


def test_dry_run_prints_the_request_and_touches_nothing_else(cli, capsys):
    aws = clients()
    args = cli.argparse.Namespace(
        preset="benchmark", run="lego-800px", scene="lego", iterations=None, max_hours=None,
        instance_type="ml.g5.xlarge", bucket=None, extra="", no_watch=True, dry_run=True)
    printed = []
    with Stubber(aws["sts"]) as sts, Stubber(aws["s3"]), Stubber(aws["sagemaker"]), Stubber(aws["iam"]):
        sts.add_response("get_caller_identity", {"Account": ACCOUNT, "UserId": "x", "Arn": "arn:aws:iam::123456789012:root"})
        code = cli.launch(args, aws, out=printed.append)   # any other AWS call would raise
    assert code == 0
    request = json.loads("\n".join(printed))
    assert request["AlgorithmSpecification"] == make_request()["AlgorithmSpecification"]
    assert request["CheckpointConfig"] == make_request()["CheckpointConfig"]
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == 45 * 60


def test_the_full_preset_demands_a_time_limit(cli):
    aws = clients()
    args = cli.argparse.Namespace(
        preset="full", run="lego-800px", scene="lego", iterations=None, max_hours=None,
        instance_type="ml.g5.xlarge", bucket=None, extra="", no_watch=True, dry_run=True)
    printed = []
    with Stubber(aws["sts"]), Stubber(aws["sagemaker"]):
        assert cli.launch(args, aws, out=printed.append) == 2
    assert "--max-hours" in printed[0]


def launch_args(cli, **overrides):
    settings = dict(
        preset="benchmark", run="lego-800px", scene="lego", iterations=None, max_hours=None,
        instance_type="ml.g5.xlarge", bucket=None, extra="", no_watch=True, dry_run=False)
    settings.update(overrides)
    return cli.argparse.Namespace(**settings)


@pytest.fixture
def frozen_clock(cli, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return WHEN

    monkeypatch.setattr(cli, "datetime", Clock)


def stub_complete_setup(cli, s3, iam, sagemaker, active=()):
    s3.add_response("list_objects_v2", {"Contents": [{"Key": "nerf/data/lego/transforms_train.json"}]})
    iam.add_response("get_role", {"Role": {
        "Path": "/", "RoleName": cli.ROLE_NAME, "RoleId": "AROAEXAMPLEEXAMPLE1",
        "Arn": f"arn:aws:iam::{ACCOUNT}:role/{cli.ROLE_NAME}", "CreateDate": WHEN}})
    iam.add_response("list_attached_role_policies", {"AttachedPolicies": [
        {"PolicyName": "AmazonSageMakerFullAccess",
         "PolicyArn": "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"}]})
    sagemaker.add_response("list_training_jobs", summaries(*active))
    sagemaker.add_response("list_training_jobs", summaries())


def test_launch_checks_then_uploads_then_starts_the_job(cli, frozen_clock):
    aws = clients()
    files = job_spec.code_bundle(ROOT)
    job = "nerf-lego-800px-20261007-172501"
    expected = make_request(tags={
        "project": "nerf-replication", "run": "lego-800px", "git-commit": cli.git_commit(ROOT)})
    printed = []
    with Stubber(aws["sts"]) as sts, Stubber(aws["s3"]) as s3, Stubber(aws["iam"]) as iam, \
            Stubber(aws["sagemaker"]) as sagemaker:
        sts.add_response("get_caller_identity", {"Account": ACCOUNT, "UserId": "x", "Arn": "arn:aws:iam::123456789012:root"})
        stub_complete_setup(cli, s3, iam, sagemaker)
        s3.add_response("list_objects_v2", {}, {                     # is there a checkpoint already?
            "Bucket": BUCKET, "Prefix": "nerf/runs/lego-800px/out/checkpoint.pt", "MaxKeys": 1})
        for relative, path in files.items():
            s3.add_response("put_object", {}, {
                "Bucket": BUCKET, "Key": f"nerf/code/{job}/{relative}", "Body": path.read_bytes()})
        sagemaker.add_response(
            "create_training_job",
            {"TrainingJobArn": f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:training-job/{job}"},
            expected)                                                # exactly the described request
        code = cli.launch(launch_args(cli), aws, out=printed.append)
        for stub in (sts, s3, iam, sagemaker):
            stub.assert_no_pending_responses()
    assert code == 0
    text = "\n".join(printed)
    assert f"Started job {job}" in text
    assert "has no checkpoint yet; this job starts it" in text
    assert "stopped automatically after 45 min" in text


def test_launch_starts_nothing_when_a_job_of_the_run_is_active(cli, frozen_clock):
    aws = clients()
    printed = []
    with Stubber(aws["sts"]) as sts, Stubber(aws["s3"]) as s3, Stubber(aws["iam"]) as iam, \
            Stubber(aws["sagemaker"]) as sagemaker:
        sts.add_response("get_caller_identity", {"Account": ACCOUNT, "UserId": "x", "Arn": "arn:aws:iam::123456789012:root"})
        stub_complete_setup(cli, s3, iam, sagemaker, active=["nerf-lego-800px-20261007-100000"])
        # No upload and no create_training_job are stubbed: either call would raise.
        code = cli.launch(launch_args(cli), aws, out=printed.append)
    assert code == 1
    assert printed[-1] == "Nothing was started."
    assert any("still active" in line for line in printed)


def test_launch_reports_a_refusal_by_aws(cli, frozen_clock):
    aws = clients()
    printed = []
    with Stubber(aws["sts"]) as sts, Stubber(aws["s3"]) as s3, Stubber(aws["iam"]) as iam, \
            Stubber(aws["sagemaker"]) as sagemaker:
        sts.add_response("get_caller_identity", {"Account": ACCOUNT, "UserId": "x", "Arn": "arn:aws:iam::123456789012:root"})
        stub_complete_setup(cli, s3, iam, sagemaker)
        s3.add_response("list_objects_v2", {"Contents": [{"Key": "nerf/runs/lego-800px/out/checkpoint.pt"}]})
        for _ in job_spec.code_bundle(ROOT):
            s3.add_response("put_object", {})
        sagemaker.add_client_error(
            "create_training_job", "ResourceLimitExceeded",
            "The account-level service limit 'ml.g5.xlarge for training job usage' is 0 Instances")
        code = cli.launch(launch_args(cli), aws, out=printed.append)
    assert code == 1
    text = "\n".join(printed)
    assert "already has a checkpoint; this job continues it" in text
    assert "ResourceLimitExceeded: The account-level service limit" in text
    assert "Nothing was started" in text


def test_time_limit_and_extra_arguments_reach_the_request(cli, frozen_clock):
    aws = clients()
    printed = []
    args = launch_args(cli, preset="full", max_hours=1.5, iterations=200_000, run="lego-400px",
                       extra="--downscale 2 --seed 3", dry_run=True)
    with Stubber(aws["sts"]) as sts:
        sts.add_response("get_caller_identity", {"Account": ACCOUNT, "UserId": "x", "Arn": "arn:aws:iam::123456789012:root"})
        assert cli.launch(args, aws, out=printed.append) == 0
    request = json.loads("\n".join(printed))
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == 5400
    arguments = request["AlgorithmSpecification"]["ContainerArguments"]
    assert arguments[arguments.index("--iterations") + 1] == "200000"
    assert arguments[arguments.index("--validate-every") + 1] == "10000"
    assert arguments[-4:] == ["--downscale", "2", "--seed", "3"]
    assert request["CheckpointConfig"]["S3Uri"].endswith("/nerf/runs/lego-400px/out")


def job_summaries(*jobs):
    return {"TrainingJobSummaries": [
        {"TrainingJobName": name, "TrainingJobArn": f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:training-job/{name}",
         "CreationTime": WHEN.replace(hour=hour), "TrainingJobStatus": "Completed"} for name, hour in jobs]}


def test_the_latest_job_is_the_most_recently_created_one_of_this_project(cli):
    aws = clients()
    with Stubber(aws["sagemaker"]) as sagemaker:
        sagemaker.add_response("list_training_jobs", job_summaries(
            ("nerf-lego-800px-20261007-100000", 10), ("nerf-sim-20261007-150000", 15),
            ("my-nerf-experiment", 20), ("nerf-lego-800px-20261007-120000", 12)))
        assert cli.latest_job(aws["sagemaker"]) == "nerf-sim-20261007-150000"
        sagemaker.add_response("list_training_jobs", job_summaries())
        assert cli.latest_job(aws["sagemaker"]) is None


def test_stop_asks_sagemaker_to_stop_the_running_job(cli):
    aws = clients()
    printed = []
    args = cli.argparse.Namespace(job=None)
    with Stubber(aws["sagemaker"]) as sagemaker:
        sagemaker.add_response("list_training_jobs", job_summaries(("nerf-lego-800px-20261007-100000", 10)),
                               {"NameContains": "nerf-", "SortBy": "CreationTime", "SortOrder": "Descending",
                                "MaxResults": 100, "StatusEquals": "InProgress"})
        sagemaker.add_response("stop_training_job", {}, {"TrainingJobName": "nerf-lego-800px-20261007-100000"})
        assert cli.stop(args, aws, out=printed.append) == 0
        sagemaker.assert_no_pending_responses()

        sagemaker.add_response("list_training_jobs", job_summaries())
        assert cli.stop(args, aws, out=printed.append) == 0          # nothing running: nothing to stop
    assert "Asked SageMaker to stop nerf-lego-800px-20261007-100000" in printed[0]
    assert printed[1] == "No training job of this project is running."


def test_status_shows_the_state_and_the_end_of_the_log(cli):
    aws = clients()
    job = "nerf-lego-800px-20261007-172501"
    printed = []
    args = cli.argparse.Namespace(job=None, watch=False, lines=3)
    with Stubber(aws["sagemaker"]) as sagemaker, Stubber(aws["logs"]) as logs:
        sagemaker.add_response("list_training_jobs", job_summaries((job, 17)))
        sagemaker.add_response("describe_training_job", description(
            job, "InProgress", "Training", TrainingStartTime=datetime.now(timezone.utc)))
        logs.add_response("describe_log_streams", {"logStreams": [{"logStreamName": f"{job}/algo-1-1"}]})
        logs.add_response("get_log_events", log_page(["step 100", "step 200", "step 300"], "f/3"), {
            "logGroupName": cli.LOG_GROUP, "logStreamName": f"{job}/algo-1-1", "limit": 3, "startFromHead": False})
        assert cli.status(args, aws, out=printed.append) == 0
    assert printed[0] == f"job      {job}"
    assert printed[1] == "status   InProgress (Training)"
    assert any(line.startswith("running  0.0 min so far") for line in printed)
    assert printed[-3:] == ["  step 100", "  step 200", "  step 300"]


def test_a_cpu_machine_gets_the_cpu_image_and_device(cli, frozen_clock):
    aws = clients()
    printed = []
    with Stubber(aws["sts"]) as sts:
        sts.add_response("get_caller_identity", {"Account": ACCOUNT, "UserId": "x", "Arn": "arn:aws:iam::123456789012:root"})
        cli.launch(launch_args(cli, instance_type="ml.m5.xlarge", dry_run=True), aws, out=printed.append)
    request = json.loads("\n".join(printed))
    arguments = request["AlgorithmSpecification"]["ContainerArguments"]
    assert arguments[arguments.index("--device") + 1] == "cpu"
    assert "-cpu-" in request["AlgorithmSpecification"]["TrainingImage"]


def test_a_job_that_is_still_stopping_also_blocks_a_new_one(cli):
    aws = clients()
    with Stubber(aws["s3"]) as s3, Stubber(aws["iam"]) as iam, Stubber(aws["sagemaker"]) as sagemaker:
        s3.add_response("list_objects_v2", {"Contents": [{"Key": "nerf/data/lego/transforms_train.json"}]})
        iam.add_response("get_role", {"Role": {
            "Path": "/", "RoleName": cli.ROLE_NAME, "RoleId": "AROAEXAMPLEEXAMPLE1",
            "Arn": f"arn:aws:iam::{ACCOUNT}:role/{cli.ROLE_NAME}", "CreateDate": WHEN}})
        iam.add_response("list_attached_role_policies", {"AttachedPolicies": [
            {"PolicyName": "AmazonSageMakerFullAccess",
             "PolicyArn": "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"}]})
        sagemaker.add_response("list_training_jobs", summaries(), {
            "NameContains": "nerf-lego-800px-", "StatusEquals": "InProgress", "MaxResults": 100})
        sagemaker.add_response("list_training_jobs", summaries("nerf-lego-800px-20261007-100000"), {
            "NameContains": "nerf-lego-800px-", "StatusEquals": "Stopping", "MaxResults": 100})
        problems = cli.preflight(aws, BUCKET, "lego", "lego-800px", out=lambda line: None)
    assert len(problems) == 1 and "still active" in problems[0]
