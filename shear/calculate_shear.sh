#!/bin/bash
#PBS -Pw85
#PBS -qnormal
#PBS -N era5shear
#PBS -m ae
#PBS -M craig.arthur@ga.gov.au
#PBS -lwalltime=5:00:00
#PBS -lmem=128GB,ncpus=16,jobfs=4000MB
#PBS -W umask=0002
#PBS -joe
#PBS -o ~/tcha/logs/calculate_shear.out.log
#PBS -e ~/tcha/logs/calculate_shear.err.log
#PBS -lstorage=gdata/w85+scratch/w85+gdata/rt52+gdata/dk92
#PBS -v NJOBS,NJOB,YEAR

# Run this with the following command line:
#
# qsub -v NJOBS=42,YEAR=1979 calculate_shear.sh
#
# This will run the process for 42 years, starting 1979

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


cd $HOME/tcha/shear

ECHO=/bin/echo

if [ X$NJOBS == X ]; then
    $ECHO "NJOBS (total number of jobs in sequence) is not set - defaulting to 1"
    export NJOBS=1
else
    if [ X$NJOB == X ]; then
    $ECHO "NJOBS set to $NJOBS"
    fi 
fi
  
if [ X$NJOB == X ]; then
    $ECHO "NJOB (current job number in sequence) is not set - defaulting to 1"
    export NJOB=1
fi

#
# Quick termination of job sequence - look for a specific file 
#
if [ -f STOP_SEQUENCE ] ; then
    $ECHO  "Terminating sequence at job number $NJOB"
    exit 0
fi

if [ X$NJOB == X1 ]; then
    $ECHO "This is the first year - it's not a restart"
    export YEAR=1979
    
else
    export YEAR=$(($YEAR+1))
fi
$ECHO "Processing 0-6km shear for $YEAR"

mpirun -np $PBS_NCPUS --mca coll ^hcoll python3 calculate_shear.py -c calculate.ini -y $YEAR 2> $HOME/tcha/logs/calculate_shear.stdout.$YEAR 2>&1

if [ $NJOB -lt $NJOBS ]; then
    NJOB=$(($NJOB+1))
    $ECHO "Submitting job number $NJOB in sequence of $NJOBS jobs"
    qsub -v NJOB=$NJOB,NJOBS=$NJOBS,YEAR=$YEAR calculate_shear.sh
else
    $ECHO "Finished last job in sequence"
fi
