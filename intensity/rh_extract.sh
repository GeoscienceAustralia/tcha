#!/bin/bash

#PBS -P w85
#PBS -qnormal
#PBS -N rh
#PBS -m abe
#PBS -M kieran.ricardo@ga.gov.au
#PBS -l walltime=3:00:00
#PBS -l mem=50GB,ncpus=1,jobfs=4000MB
#PBS -W umask=0002
#PBS -joe
#PBS -l storage=gdata/v10+gdata/rt52

module purge

module use /g/data/v10/public/modules/modulefiles
module load dea/20210527

cd $HOME/tcha/intensity
export OMP_NUM_THREADS=1

python3 rh_extract.py