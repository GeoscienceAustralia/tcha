#!/bin/bash
#PBS -Pw85
#PBS -qhugemem
#PBS -N calcenvflow
#PBS -m ae
#PBS -M craig.arthur@ga.gov.au
#PBS -lwalltime=6:00:00
#PBS -lmem=1470GB,ncpus=48,jobfs=4000MB
#PBS -W umask=0022
#PBS -v NJOBS,NJOB,YEAR
#PBS -joe
#PBS -o /home/547/cxa547/tcha/synoptic-flow/calcenvflow.out.log
#PBS -e /home/547/cxa547/tcha/synoptic-flow/calcenvflow.err.log
#PBS -lstorage=gdata/w85+gdata/rt52+gdata/hh5+scratch/w85

# Run this with the following command line:
#
# qsub -v NJOBS=42,YEAR=1981 calc_envflow.sh
#
# This will run the process for 42 years, starting 1981

module purge
module load pbs
module load dot

module use /g/data/hh5/public/modules
module load conda/analysis3

cd $HOME/tcha/synoptic-flow

export PYTHONPATH=$PYTHONPATH:/scratch/$PROJECT/$USER/python/lib/python3.10/site-packages

umask 0022

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
    $ECHO "This is the first year - $YEAR - it's not a restart"
    export YEAR=$YEAR

else
    export YEAR=$(($YEAR+1))
fi
$ECHO "Processing envflow for $YEAR"

cd $HOME/tcha/synoptic-flow

mpiexec -n 44 python3 envflow.py $YEAR > envflow.$YEAR.log 2>&1

if [ $NJOB -lt $NJOBS ]; then
    NJOB=$(($NJOB+1))
    $ECHO "Submitting job number $NJOB in sequence of $NJOBS jobs"
    qsub -v NJOB=$NJOB,NJOBS=$NJOBS,YEAR=$YEAR calc_envflow.sh
else
    $ECHO "Finished last job in sequence"
fi
