# nerf-replication

A from-scratch PyTorch replication of *NeRF: Representing Scenes as Neural
Radiance Fields for View Synthesis* (Mildenhall et al., ECCV 2020,
[arXiv:2003.08934](https://arxiv.org/abs/2003.08934)).

The goal is to reimplement the method without copying the reference code,
train it on the paper's synthetic Lego scene, and compare my numbers with the
published ones.

## Status

In progress. The pipeline trains end to end on a small synthetic scene, and
a first short run on the paper's Lego scene at 1/8 resolution reaches 22.5 dB
on a held-out view after 2,000 steps on a laptop GPU. The full-resolution run
has not been done yet, so there is nothing to compare with the paper.

| Part | Paper | Status |
| --- | --- | --- |
| Camera rays | Section 4 | Implemented, tested |
| Positional encoding | Section 5.1, Eq. 4 | Implemented, tested |
| Stratified sampling | Section 4, Eq. 2 | Implemented, tested |
| Volume rendering | Section 4, Eqs. 1 and 3 | Implemented, tested |
| Network | Section 3, Fig. 7 | Implemented, tested |
| Hierarchical sampling | Section 5.2 | Implemented, tested |
| Training loop | Section 5.3, Eq. 6 | Implemented, tested on an analytic scene |
| Loader for the paper's synthetic scenes, and a camera check | Section 6.1 | Implemented, tested; Lego passes |
| Training script with checkpoints and exact resume | | Implemented, tested |
| Launcher for Amazon SageMaker training jobs | | Implemented, tested offline; no job run yet |
| Lego scene: PSNR, SSIM, LPIPS against the paper | Section 6 | Next |
| Mesh extraction from the trained density | not in the paper | Planned |

## First trained result

`scripts/00_smoke_test.py` trains a small network (4 layers of 64 units) for
2000 steps on 24 synthetic 48x48 images of four coloured spheres, then renders
four viewpoints that were not in the training set.

![Top row: true held-out views. Bottom row: renders.](results/smoke_test.png)

Top row: the true held-out views. Bottom row: the same views rendered by the
trained network. The numbers from the run are in `results/smoke_test.json`.
The script fails if the mean held-out PSNR is below 20 dB; a blank white image
scores about 8.5 dB.

This shows that the pipeline learns a 3D scene from 2D images alone. It is not
a replication result. The training images come from this repository's own
renderer applied to a scene with known density and colour, and the network and
sample counts are far smaller than the paper's so that the run takes a couple
of minutes on a laptop CPU.

## Data

The paper's synthetic Lego scene comes from the example archive that the
authors' own download script fetches:

```
mkdir -p data
curl -L -o data/nerf_example_data.zip http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
unzip -q data/nerf_example_data.zip -d data
python scripts/01_check_dataset.py data/nerf_synthetic/lego
```

The last command trains nothing. It checks that the dataset's cameras mean
what this code assumes: that every camera looks at the origin down its own -z
axis with world +z up, and that carving a grid with the silhouettes of all
training views leaves a visual hull which projects back onto those
silhouettes. It writes its numbers to `results/dataset_check.json`.

On Lego every check passes. All 100 training cameras are 4.0311 from the
origin and look at it to within 0.04 degrees. The hull spans x from -0.60 to
0.60, y from -1.10 to 1.12 and z from -0.51 to 0.98, and it projects back onto
the silhouettes with a mean intersection over union of 0.935 (worst view
0.916). For comparison, on the analytic test scene mirrored or upside-down
images score about 0.7, and the check requires 0.8.

## Training

```
python scripts/02_train.py data/nerf_synthetic/lego --out runs/lego_100px --downscale 8 --iterations 1000
```

Without `--downscale` and `--iterations` this is the released paper
configuration: 800x800 images, 1024 rays per step, 64 coarse and 128 fine
samples per ray, 500k steps. The run directory receives the settings, a log,
a validation image and PSNR at intervals, and a checkpoint. Running the same
command again resumes from the checkpoint, and a stop request (Ctrl-C, or the
termination signal of a job scheduler) lets the current step finish and saves
first, so an interrupted run follows exactly the same path as an uninterrupted
one. Run directories are not committed.

### As an Amazon SageMaker training job

At about 1.4 steps per second on a laptop GPU, 500k steps take four days.
`scripts/03_sagemaker.py` runs the same script on a rented GPU machine:

```
python3 scripts/03_sagemaker.py launch benchmark
python3 scripts/03_sagemaker.py launch full --max-hours 24
python3 scripts/03_sagemaker.py status
python3 scripts/03_sagemaker.py stop
```

`benchmark` is the first 2,000 steps of the paper configuration, to see that
the job runs and how fast. `full` is all 500k steps. The job is described in
`src/nerf/sagemaker.py`:

- **Container.** AWS's prebuilt PyTorch 2.10 training image. Nothing is
  installed when the job starts.
- **Program.** `python3 -u 02_train.py` with the same arguments as a local
  run, plus `--device cuda`, so that a job which cannot see its GPU fails at
  once.
- **Code.** The script and the `nerf` package are uploaded to S3 for each job
  and mounted as an input channel. The script sits next to the package, so
  nothing has to be installed. The upload is the record of what the job ran,
  and the job is tagged with the git commit.
- **Data.** The scene directory in S3 is a second input channel.
- **Run directory.** The script's `--out` directory is the job's checkpoint
  directory, which SageMaker copies to S3 while the job runs and restores when
  a later job names the same location. Jobs with the same `--run` name
  therefore continue one another: `full` picks up where `benchmark` ended, and
  a job that was stopped or reached its time limit is continued by launching
  it again.
- **Cost limit.** Every job has a time limit, after which SageMaker stops it.
  The stop signal reaches the script directly, which finishes its step and
  saves a checkpoint.

One-time setup, in region us-east-1, from a machine with the AWS CLI and the
scene:

```
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=sagemaker-us-east-1-$ACCOUNT
aws s3 mb s3://$BUCKET --region us-east-1
aws s3 sync data/nerf_synthetic/lego s3://$BUCKET/nerf/data/lego --only-show-errors
aws iam create-role --role-name nerf-sagemaker-role --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name nerf-sagemaker-role --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

The bucket name has to contain "sagemaker": the `AmazonSageMakerFullAccess`
policy lets the job read and write objects only in buckets named that way.
The launcher itself needs `boto3` and AWS credentials but not PyTorch. AWS
CloudShell, the terminal in the AWS console, has both.

No job has been run on AWS yet. The measured speed and cost will go here.

## What is here

- `src/nerf/rays.py`: the camera model. One ray per pixel from a camera pose,
  and the reverse, from a 3D point to the pixel it appears at.
- `src/nerf/encoding.py`: positional encoding of positions and directions.
- `src/nerf/model.py`: the network, from position and viewing direction to
  density and colour.
- `src/nerf/render.py`: stratified sampling, the quadrature of the volume
  rendering integral, importance sampling from the coarse weights, the
  two-pass coarse-then-fine rendering of a batch of rays, and rendering of
  whole images.
- `src/nerf/data.py`: the container for posed images, the loader for the
  paper's synthetic scenes, camera pose helpers and the analytic sphere scene.
- `src/nerf/train.py`: the training step and the state it changes,
  hyperparameters, checkpoints and evaluation by PSNR.
- `src/nerf/hull.py`: space carving. The visual hull of a scene from its
  silhouettes, used to check cameras without training.
- `scripts/00_smoke_test.py`: the end-to-end run described above.
- `scripts/01_check_dataset.py`: the dataset check described above.
- `scripts/02_train.py`: training on a synthetic scene, as described above.
- `src/nerf/sagemaker.py`: the description of a SageMaker training job for
  that script. It makes no AWS calls.
- `scripts/03_sagemaker.py`: uploads the code, starts the job, follows its
  log, stops it.
- `tests/`: unit tests for the eight modules and the launcher.

## How it is checked

**Renderer.** The rendering integral is the emission-absorption case of
radiative transfer, so the renderer can be tested against closed-form answers
and not only for tensor shapes:

- a uniform slab crossed at an angle absorbs exactly `1 - exp(-sigma * path)`
  (Beer-Lambert);
- for a density that rises linearly along the ray, the transmitted fraction
  converges to the analytic `exp(-tau)`, with the error falling by about four
  each time the number of samples is quadrupled, as expected of a first-order
  rule;
- for random densities, sample positions and ray lengths, the weights sum to
  `1 - exp(-optical depth)`;
- an opaque surface returns its own colour and depth, and nothing behind it
  contributes;
- empty space renders as the background.

**Network.** The layer sizes are those of Fig. 7, which add up to 595,844
parameters per network. Density does not change when the viewing direction
does, and colour does.

**Hierarchical sampling.** The network is replaced by an analytic opaque wall,
so the right answer is known:

- samples drawn from a set of weights occur with frequencies proportional to
  those weights, and are uniform inside each bin;
- the importance samples land in the bin where the coarse pass found the wall,
  and the fine pass puts the surface closer to its true depth than the coarse
  pass does;
- a loss on the fine render sends no gradient into the coarse network.

**Training loop.**

- every ray is paired with the colour of its own pixel, checked on a
  non-square image whose colours encode pixel coordinates;
- two steps of the loop end at the same weights as two steps written out by
  hand, and both networks are updated;
- putting the exact analytic scene where the trained networks would go, the
  evaluation code gives back the training images, which ties together the
  poses, focal length, bounds and background used at evaluation time;
- the loss falls over a short run;
- a run restored from a checkpoint ends at the same weights, with the same
  losses on the way, as a run that was never stopped;
- a checkpoint is refused if the model or sampling settings differ from the
  ones it was made with, and a save that dies halfway leaves the previous
  checkpoint intact.

**Data and cameras.**

- a scene written to disk in the dataset's format is read back exactly:
  poses, focal length, straight-alpha compositing over white, and
  area-averaged downscaling done on RGBA before compositing;
- a point on the ray through a pixel projects back to that pixel;
- on the analytic scene the visual hull contains the interior of every sphere
  and projects back onto the silhouettes with an intersection over union
  above 0.95 in every view;
- deliberately wrong cameras are caught: inverted poses or OpenCV-style axes
  leave an empty hull, and mirrored or upside-down images drop the
  intersection over union below 0.8.

**SageMaker job.** These run without an AWS account:

- the request satisfies every length, range, pattern and enumeration rule in
  the description of the API that ships with the AWS SDK. The SDK itself
  checks only types and required fields before sending;
- the job's exact command line, run on a copy of the uploaded files in an
  empty directory, trains, and run again with a larger step count it continues
  from the checkpoint. A marker written on import shows that the uploaded
  package was the one used;
- the launcher is run against canned AWS responses: it reports every missing
  piece of the setup, refuses to start a second job of a run that is still
  active, and when following a job prints each log line once.

## Where this follows the released code and not the paper

The paper and the authors' code ([bmild/nerf](https://github.com/bmild/nerf))
disagree in a few places. Since the published numbers came from the code, I
follow the code:

1. **Positional encoding.** Eq. 4 has `sin(2^k pi p)`. The code uses
   `sin(2^k p)` with no factor of pi, and also concatenates the raw input.
2. **Stratified sampling.** Eq. 2 splits the ray into N equal bins. The code
   places N evenly spaced points, including both ends, and jitters each one
   between the midpoints to its neighbours, so the two end bins are half width.
3. **Ray parameter.** Ray directions are not normalised. `t` measures depth in
   front of the camera, so near and far are planes, and the renderer multiplies
   by the length of the direction to get distances.
4. **Last sample.** The final sample on each ray is given an effectively
   infinite interval, so any density there is fully opaque.
5. **Hierarchical sampling.** The importance samples are drawn from bins whose
   edges are the midpoints between coarse samples, using the weights of the
   interior coarse samples only. The first and last coarse weights are dropped.
6. **Batch size.** Section 5.3 says 4096 rays per step. The released
   configuration for the paper's synthetic results
   (`paper_configs/blender_config.txt`) uses 1024.
7. **Training length.** Section 5.3 says 100-300k iterations. The same
   configuration file says it reproduces the Lego result when trained to 500k
   iterations, and its learning rate falls by a factor of 10 every 500k steps,
   which gives the paper's 5e-4 to 5e-5 only over a 500k run.

One more choice is inherited from the code and is not stated in the paper:
the network is initialised Glorot-uniform with zero biases, the default of the
TensorFlow layers the authors used. PyTorch's default is different.

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
python scripts/00_smoke_test.py
```

On an Apple Silicon Mac, create the environment from a native arm64 Python.
An Intel build of Python running under Rosetta can only install PyTorch 2.2.2,
the last release with Intel macOS wheels, and that release predates NumPy 2.

## Reference

B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi and
R. Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.
ECCV 2020.
