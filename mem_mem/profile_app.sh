#!/bin/bash

for (( N=12000; N<=300000; N+=1000 ))
do

nvprof --print-gpu-trace --csv ./mem_mem_ptx_all 2 0 $N 0 2> ./profile_results/trace_$N.csv

nvprof --metrics all --csv ./mem_mem_ptx_all 2 0 $N 0 2> ./profile_results/metrics_$N.csv

done
