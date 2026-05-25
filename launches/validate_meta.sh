#!/usr/bin/bash -l
#SBATCH --job-name=validate_meta
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=A100:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=out/validate_meta.out
#SBATCH --error=out/validate_meta.out

module load multigpu cuda/12.6
source .venv/bin/activate
srun python -m tamart.experiments.validate_meta --batch-size 256
