#!/bin/bash

for (( N=40000; N<=52000; N+=200 ))
do

nvprof --print-gpu-trace --csv ./cmp_cmp_ptx 2 0 $N 0 2> ./profile_results/trace_$N.csv

nvprof --metrics all --csv ./cmp_cmp_ptx 2 0 $N 0 2> ./profile_results/metrics_$N.csv

done
