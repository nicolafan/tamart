#!/usr/bin/bash -l
#SBATCH --job-name=classify
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=A100:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=out/classify.out
#SBATCH --error=out/classify.out

module load multigpu cuda/12.6
source .venv/bin/activate
srun python -m tamart.experiments.classify --batch-size 256
