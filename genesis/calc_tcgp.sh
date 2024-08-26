#!/bin/bash
#PBS -Pw85
#PBS -qnormal
#PBS -N tcgp
#PBS -m ae
#PBS -M craig.arthur@ga.gov.au
#PBS -lwalltime=4:00:00
#PBS -lmem=160GB,ncpus=32,jobfs=4000MB
#PBS -W umask=0022
#PBS -joe
#PBS -o /home/547/cxa547/pcmin/logs/tcgp.out.log
#PBS -e /home/547/cxa547/pcmin/logs/tcgp.err.log
#PBS -lstorage=gdata/w85+gdata/rt52+gdata/hh5+scratch/w85

module use /g/data/hh5/public/modules
module load conda/analysis3

export PYTHONPATH=/scratch/w85/cxa547/python/lib/python3.10/site-packages:$PYTHONPATH

cd $HOME/tcha/genesis

python calculate_genesis_regions.py -c calculate.ini > $HOME/pcmin/logs/calculate_tcgp.stdout 2>&1

