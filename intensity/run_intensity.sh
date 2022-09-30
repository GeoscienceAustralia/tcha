#!/bin/bash

#PBS -P w85
#PBS -qnormal
#PBS -N runIntensity
#PBS -m abe
#PBS -M craig.arthur@ga.gov.au
#PBS -l walltime=2:30:00
#PBS -l mem=96GB,ncpus=48,jobfs=4000MB
#PBS -W umask=0002
#PBS -joe
#PBS -l storage=gdata/hh5+gdata/w85
module use /g/data/hh5/public/modules
module load conda/analysis3

cd ~/tcha/intensity

python tc_intensity_analysis.py
