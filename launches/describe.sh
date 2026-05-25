#!/usr/bin/bash -l
#SBATCH --job-name=describe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=A100:1
#SBATCH --mem=32G
#SBATCH --time=7:00:00
#SBATCH --output=out/describe.out
#SBATCH --error=out/describe.out

module load multigpu cuda/12.6
source .venv/bin/activate
srun python -m tamart.experiments.describe --batch-size 4
