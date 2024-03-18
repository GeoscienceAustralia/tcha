#!/bin/bash
#PBS -Pw85
#PBS -qhugemem
#PBS -N envflowbatch
#PBS -m ae
#PBS -M craig.arthur@ga.gov.au
#PBS -lwalltime=24:00:00
#PBS -lmem=1470GB,ncpus=48,jobfs=4000MB
#PBS -joe
#PBS -W umask=002
#PBS -lstorage=gdata/w85+scratch/w85+gdata/hh5+gdata/rt52

umask 0002

module purge
module load pbs
module load dot
module use /g/data/hh5/public/modules
module load conda/analysis3

cd $HOME/tcha/synoptic-flow
export PYTHONPATH=$PYTHONPATH:/scratch/$PROJECT/$USER/python/lib/python3.10/site-packages

mpirun -np $PBS_NCPUS python3 envflow.py $YEAR > envflow.batch.log 2>&1

