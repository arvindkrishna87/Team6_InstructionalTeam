#!/bin/bash
#SBATCH --account=e32998                                  # groups command on Quest to find out
#SBATCH --partition=short                                  # short, normal, long
#SBATCH --time=10:00                                  # 10 minutes
#SBATCH --nodes=1                                         # recommended is 1 node
#SBATCH --ntasks=1                                        # number of cores, this can be increased as needed but not too much
#SBATCH --mem=1G                                         # memory per node
#SBATCH --output=test.out                                 # output file
#SBATCH --error=test.err                                  # error file
#SBATCH --job-name=test                                   # job name
#SBATCH --mail-type=ALL                                   # send email at begin and end of job or failure
#SBATCH --mail-user="krish@northwestern.edu"          # (change this) your NU email address

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

python3 print_name.py
