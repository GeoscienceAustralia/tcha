#!/bin/bash

#PBS -Pw85
#PBS -qnormal
#PBS -N era5dlm
#PBS -m abe
#PBS -M kieran.ricardo@ga.gov.au
#PBS -lwalltime=1:00:00
#PBS -lmem=128GB,ncpus=48,jobfs=4000MB
#PBS -W umask=0002
#PBS -joe
#PBS -lstorage=gdata/w85+scratch/w85+gdata/rt52

module purge
module load pbs
module load dot

module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-data-analysis/2021.09
module load openmpi/4.1.0

# This is where mpi4py is installed:
export PYTHONPATH=$PYTHONPATH:/g/data/w85/.local/lib/python3.8/site-packages/

# flStartLog, attemptParallel, etc.
export PYTHONPATH=$PYTHONPATH:$HOME/pylib/python

cd $HOME/tcha/synoptic-flow

mpiexec -n 48 python era_5_extract.py
