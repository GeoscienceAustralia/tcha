#!/bin/bash
#PBS -Pw85
#PBS -qnormal
#PBS -N envflowbatch
#PBS -m ae
#PBS -M craig.arthur@ga.gov.au
#PBS -lwalltime=24:00:00
#PBS -lmem=190GB,ncpus=48,jobfs=4000MB
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


# With all these scripts, check that the basins align - e.g. "SH". In some scripts, this
# is set inline, not by a command line argument, so check each carefully.
mpirun -np $PBS_NCPUS python3 envflow.py > $HOME/tcha/logs/envflow.SH.log 2>&1

# Run the BAM parameter fitting. This script prints info to stdout, so we
# direct the output to a different location so we can review:
python3 fitBAMparameters.py > /scratch/$PROJECT/$USER/envflow/SH/fitBAM.log 2>&1

# Run the vorticity analysis:
python3 vorticity_analysis.py > $HOME/tcha/logs/vorticity_analysis.SH.log 2>&1

# Run beta drift calculations. This script is 
python3 plotBetaDrift.py > $HOME/tcha/logs/betadrift.SH.log 2>&1
