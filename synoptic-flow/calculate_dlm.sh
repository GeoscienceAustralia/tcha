#!/bin/bash

#PBS -Pw85
#PBS -qnormal
#PBS -N era5shear
#PBS -m ae
#PBS -M kieran.ricardo@ga.gov.au
#PBS -lwalltime=1:00:00
#PBS -lmem=128GB,ncpus=48,jobfs=4000MB
#PBS -W umask=0002
#PBS -joe
#PBS -lstorage=gdata/w85+scratch/w85+gdata/rt52
#PBS -v NJOBS,NJOB,YEAR

mpiexec -n 48 python era_5_extract.py
