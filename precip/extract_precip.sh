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
#PBS -lstorage=gdata/w85+scratch/w85+gdata/rt52+gdata/hh5

module use module use /g/data/hh5/public/modules


module load conda/analysis3
export PYTHONPATH=$PYTHONPATH:$HOME/pylib/python
conda activate analysis3
cd $HOME/tcha/precip
python extract_precip.py 2>&1 extract_precip.stdout
