#!/bin/sh
#PBS -Pw85
#PBS -q hugemem
#PBS -N extractera5
#PBS -m ae
#PBS -M craig.arthur@ga.gov.au
#PBS -lwalltime=6:00:00
#PBS -lmem=1470GB,ncpus=48,jobfs=400MB
#PBS -joe
#PBS -lstorage=gdata/w85+scratch/w85+gdata/hh5+gdata/rt52
#PBS -l wd
#PBS -W umask=0002

module use /g/data/hh5/public/modules
module load conda/analysis3

# For locally installed packages [cdsapi, global_land_mask]
export PYTHONPATH=$PYTHONPATH:/scratch/$PROJECT/$USER/python/lib/python3.10/site-packages

cd $HOME/tcha/synoptic-flow
mpiexec -n $PBS_NCPUS python3 extract_era5.py
