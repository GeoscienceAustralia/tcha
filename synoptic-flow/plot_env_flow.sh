#!/bin/bash

#PBS -Pw85
#PBS -qhugemem
#PBS -N plotflow
#PBS -m abe
#PBS -M craig.arthur@ga.gov.au
#PBS -l walltime=5:00:00
#PBS -l mem=1460GB,ncpus=48,jobfs=4000MB
#PBS -W umask=0002
#PBS -joe
#PBS -lstorage=gdata/w85+scratch/w85+gdata/rt52+gdata/hh5

module purge
module use /g/data/hh5/public/modules
module load conda/analysis3
export PYTHONPATH=$PYTHONPATH:/scratch/$PROJECT/$USER/python/lib/python3.10/site-packages

cd $HOME/tcha/synoptic-flow

mpiexec -n 44 python3 plot_environmental_flow.py
