#!/bin/bash
#PBS -Pw85
#PBS -qnormal
#PBS -N tcprecip
#PBS -m ae
#PBS -M craig.arthur@ga.gov.au
#PBS -lwalltime=3:00:00
#PBS -lmem=32GB,ncpus=16,jobfs=4000MB
#PBS -W umask=0002
#PBS -joe
#PBS -lstorage=gdata/w85+scratch/w85+gdata/rt52+gdata/dk92

module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-data-analysis/2021.09
export PYTHONPATH=$PYTHONPATH:$HOME/pylib/python

cd $HOME/tcha/precip
python3 extract_precip.py 2>&1 extract_precip.stdout
