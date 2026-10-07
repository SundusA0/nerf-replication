"""Describing a training run as an Amazon SageMaker training job.

A training job is a container that SageMaker starts on a rented machine, feeds
with files from S3, and tears down when the program inside it exits. This
module builds the description of such a job for `scripts/02_train.py`. It makes
no AWS calls and does not import the AWS SDK, so everything in it is tested
offline; `scripts/03_sagemaker.py` uploads the code and sends the request.

How a local run maps onto a job:

  program    The container runs `python3 -u 02_train.py <arguments>`, the same
             script and arguments as on a laptop. Nothing is installed at run
             time: the container is AWS's prebuilt PyTorch image.
  code       The script and the `nerf` package are uploaded to S3 and mounted
             in the container as an input channel. The script sits next to the
             package, so Python finds `nerf` without it being installed.
  data       The scene directory is a second input channel.
  run        The script's --out directory is the job's checkpoint directory.
             SageMaker copies that directory to S3 while the job runs, and
             restores it when a later job names the same S3 location. A job
             that was stopped, or that hit its time limit, is therefore
             continued by launching it again, exactly like running the same
             command again locally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REGION = "us-east-1"

# AWS Deep Learning Containers, PyTorch 2.10 for SageMaker training. The tags
# are the ones listed in github.com/aws/deep-learning-containers. The registry
# account is specific to the region.
IMAGE_REGISTRY = "763104351884.dkr.ecr.us-east-1.amazonaws.com"
IMAGE_TAGS = {
    "gpu": "2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker",
    "cpu": "2.10.0-cpu-py313-ubuntu22.04-sagemaker",
}

# Where SageMaker puts things inside the container.
CODE_DIR = "/opt/ml/input/data/code"      # input channel "code"
DATA_DIR = "/opt/ml/input/data/training"  # input channel "training"
RUN_DIR = "/opt/ml/checkpoints"           # copied to S3 during the job

ENTRY_SCRIPT = "02_train.py"

# Limits of the CreateTrainingJob API that this module can run into.
MAX_JOB_NAME_LENGTH = 63
JOB_NAME_PATTERN = re.compile(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}")
MAX_ARGUMENTS = 100
MAX_ARGUMENT_LENGTH = 256
RUN_NAME_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


@dataclass(frozen=True)
class Preset:
    """How long to train and how often to report, for one kind of job."""

    iterations: int
    validate_every: int
    checkpoint_every: int
    max_runtime_minutes: int | None  # None: must be chosen when launching


PRESETS = {
    # A few minutes of the real configuration: checks that the job runs at all
    # and measures its speed. It is the start of the full run, not a throwaway:
    # launched under the same run name, "full" continues from its checkpoint.
    "benchmark": Preset(
        iterations=2_000, validate_every=1_000, checkpoint_every=500, max_runtime_minutes=45
    ),
    # The released paper configuration.
    "full": Preset(
        iterations=500_000, validate_every=10_000, checkpoint_every=1_000, max_runtime_minutes=None
    ),
}


def is_gpu_instance(instance_type: str) -> bool:
    """Whether a SageMaker instance type such as "ml.g5.xlarge" has a GPU.

    The G and P families are the GPU families used for training.
    """
    parts = instance_type.split(".")
    if len(parts) != 3 or parts[0] != "ml" or not all(parts):
        raise ValueError(f"not a SageMaker instance type: {instance_type!r}")
    return parts[1][0] in ("g", "p")


def image_uri(instance_type: str) -> str:
    """The container image for an instance type: the GPU or the CPU build."""
    tag = IMAGE_TAGS["gpu" if is_gpu_instance(instance_type) else "cpu"]
    return f"{IMAGE_REGISTRY}/pytorch-training:{tag}"


def make_job_name(run: str, now: datetime) -> str:
    """A job name unique to the second: nerf-<run>-<YYYYMMDD>-<HHMMSS>.

    Job names can never be reused within an account, so the launch time is part
    of the name. The run name is what ties several jobs to one training run.
    """
    if not RUN_NAME_PATTERN.fullmatch(run):
        raise ValueError(
            f"run name {run!r} must be lower-case letters and digits separated by single hyphens"
        )
    name = f"nerf-{run}-{now:%Y%m%d-%H%M%S}"
    if len(name) > MAX_JOB_NAME_LENGTH:
        raise ValueError(f"run name {run!r} is too long: job name {name!r} exceeds 63 characters")
    return name


def s3_layout(bucket: str, run: str, job_name: str, scene: str) -> dict[str, str]:
    """Where everything lives in the bucket.

    Input locations end in a slash and the two output locations do not; those
    are the forms the API documentation gives for each field.
    """
    base = f"s3://{bucket}/nerf"
    return {
        "data": f"{base}/data/{scene}/",       # the scene, uploaded once
        "code": f"{base}/code/{job_name}/",    # the code exactly as this job ran it
        "run": f"{base}/runs/{run}/out",       # the run directory, shared by the jobs of a run
        "output": f"{base}/output",            # required by the API; the script does not use it
    }


def code_bundle(repo_root: Path) -> dict[str, Path]:
    """Files a job needs, keyed by their path inside the container's code directory.

    The training script is placed next to the `nerf` package. A script's own
    directory is on Python's import path, so `from nerf... import ...` works in
    the container with no installation step and no environment variables.
    """
    repo_root = Path(repo_root)
    files = {ENTRY_SCRIPT: repo_root / "scripts" / ENTRY_SCRIPT}
    for path in sorted((repo_root / "src" / "nerf").glob("*.py")):
        files[f"nerf/{path.name}"] = path
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing or "nerf/train.py" not in files:
        raise FileNotFoundError(f"not a checkout of this repository: {repo_root} ({missing})")
    return files


def training_arguments(
    iterations: int,
    validate_every: int,
    checkpoint_every: int,
    device: str,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """Command-line arguments for `02_train.py` inside the container.

    The device is always stated. On a GPU machine it is "cuda", so that a job
    which cannot see its GPU fails at once and does not train on the CPU for
    hours at GPU prices.
    """
    return [
        DATA_DIR,
        "--out", RUN_DIR,
        "--iterations", str(iterations),
        "--validate-every", str(validate_every),
        "--checkpoint-every", str(checkpoint_every),
        "--device", device,
        *extra,
    ]


def training_job_request(
    *,
    job_name: str,
    run: str,
    bucket: str,
    role_arn: str,
    instance_type: str,
    max_runtime_seconds: int,
    arguments: list[str],
    scene: str = "lego",
    tags: dict[str, str] | None = None,
) -> dict:
    """The CreateTrainingJob request for one job.

    Args:
        job_name: from `make_job_name`.
        run: name of the training run this job belongs to. Jobs with the same
            run name share a run directory in S3 and continue one another.
        bucket: S3 bucket holding the data and receiving the results.
        role_arn: IAM role the job runs as. It needs to read and write the
            bucket, pull the image and write logs.
        instance_type: e.g. "ml.g5.xlarge" (one NVIDIA A10G).
        max_runtime_seconds: SageMaker stops the job after this long. This is
            the hard limit on what a job can cost.
        arguments: from `training_arguments`.
        scene: name of the scene directory under nerf/data/ in the bucket.
        tags: attached to the job, e.g. the git commit of the code.
    """
    where = s3_layout(bucket, run, job_name, scene)

    def channel(name: str, s3_uri: str) -> dict:
        return {
            "ChannelName": name,
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_uri,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "InputMode": "File",  # download everything before the program starts
        }

    request = {
        "TrainingJobName": job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri(instance_type),
            "TrainingInputMode": "File",
            # -u: unbuffered output, so log lines appear as they are printed.
            # python3 is the program itself, not a child of a shell, so the
            # stop signal SageMaker sends reaches the script, which finishes
            # its current step and saves a checkpoint before exiting.
            "ContainerEntrypoint": ["python3", "-u", f"{CODE_DIR}/{ENTRY_SCRIPT}"],
            "ContainerArguments": list(arguments),
        },
        "InputDataConfig": [channel("training", where["data"]), channel("code", where["code"])],
        "OutputDataConfig": {"S3OutputPath": where["output"]},
        "CheckpointConfig": {"S3Uri": where["run"], "LocalPath": RUN_DIR},
        "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 30},
        "StoppingCondition": {"MaxRuntimeInSeconds": int(max_runtime_seconds)},
        "Tags": [{"Key": key, "Value": value} for key, value in sorted((tags or {}).items())],
    }
    check_request(request)
    return request


def check_request(request: dict) -> None:
    """Raise ValueError if the request breaks an API limit this module can reach.

    The AWS SDK checks types and required fields before sending, but not
    lengths, so those are checked here.
    """
    name = request["TrainingJobName"]
    if not JOB_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid training job name: {name!r}")

    spec = request["AlgorithmSpecification"]
    for field in ("ContainerEntrypoint", "ContainerArguments"):
        items = spec[field]
        if not 1 <= len(items) <= MAX_ARGUMENTS:
            raise ValueError(f"{field} must have 1 to {MAX_ARGUMENTS} items, not {len(items)}")
        for item in items:
            if not isinstance(item, str) or len(item) > MAX_ARGUMENT_LENGTH:
                raise ValueError(f"{field} item is not a string of at most 256 characters: {item!r}")

    if request["StoppingCondition"]["MaxRuntimeInSeconds"] < 1:
        raise ValueError("the time limit must be positive")
