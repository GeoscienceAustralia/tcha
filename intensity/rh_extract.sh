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
#PBS -l storage=gdata/hh5+gdata/rt52+gdata/w85

module purge

module use /g/data/hh5/public/modules
module load conda/analysis3

cd $HOME/tcha/intensity
export OMP_NUM_THREADS=1

python3 rh_extract.py