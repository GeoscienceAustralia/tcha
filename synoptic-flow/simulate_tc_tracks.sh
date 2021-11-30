#!/bin/bash

#PBS -Pw85
#PBS -qnormal
#PBS -N era5dlm
#PBS -m abe
#PBS -M kieran.ricardo@ga.gov.au
#PBS -lwalltime=5:00:00
#PBS -lmem=128GB,ncpus=48,jobfs=4000MB
#PBS -W umask=0002
#PBS -joe
#PBS -lstorage=gdata/w85+scratch/w85+gdata/rt52

module purge
module use /g/data/v10/public/modules/modulefiles
module load dea/20210527

cd $HOME/tcha/synoptic-flow

mpiexec -n 24 python3 simulate_tc_tracks.py
