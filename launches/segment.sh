#!/usr/bin/bash -l
#SBATCH --job-name=segment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=A100:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=out/segment.out
#SBATCH --error=out/segment.out

module load multigpu cuda/12.6
# SAM 3 needs transformers>=5, which conflicts with the vllm-pinned main .venv.
# Use the dedicated SAM 3 environment instead.
source .venv-sam3/bin/activate
# facebook/sam3 is a gated repo; tamart repoints HF_HOME to data/hf (no token
# there), so expose the user's token explicitly for the download.
export HF_TOKEN=$(cat ~/data/.cache/huggingface/token)
srun python -m tamart.experiments.segment
