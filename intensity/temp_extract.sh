#!/bin/bash

#PBS -P w85
#PBS -qnormal
#PBS -N temp_extract
#PBS -m abe
#PBS -M kieran.ricardo@ga.gov.au
#PBS -l walltime=1:30:00
#PBS -l mem=96GB,ncpus=48,jobfs=4000MB
#PBS -W umask=0002
#PBS -joe
#PBS -l storage=gdata/v10+gdata/gb6

module purge

module use /g/data/v10/public/modules/modulefiles
module load dea/20210527

cd $HOME/tcha/intensity
DATE=`date +%Y%m%d%H%M`
mpiexec -n 48 python3 temp_extract.py > temp_extract_$DATE.log 2>&1