#!/bin/bash

#SBATCH --job-name=anomaly_detection
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --mem=50gb
#SBATCH --export=ALL 
#SBATCH --time=1-00:00:00 

#SBATCH --output=main-%j.out
#SBATCH --error=main-%j.err

cd ..

module load gnu10
module load graphviz


## Activate the python virtual environment
source activate ./.venv/bin/activate

## Execute your script
python anomaly_detection/main.py