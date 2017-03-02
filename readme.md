## Author
Leiming Yu
* Email: ylm@ece.neu.edu
* Twitter: @yu_leiming
* Blog: http://www1.coe.neu.edu/~ylm/

## Motivation
Analyze the concurrent kernel execution.

## Folders
* mem_mem : memory bound kernel 
* cmp_cmp : compute bound kernel

## Techniques
* inline ptx for thread timing and other trace info
* using unified memory to store the trace
* when dumpying the trace info (for all the gpu threads), the clock() is normalized by 1e-6.


## Examples
The commandline option 

./mem_mem_ptx_all stream_num device_id data_size print_log

To run without printing log file
```bash
./mem_mem_ptx_all 1 0 1000 0
```
## Device Info
GTX 950: maxwell, 6 SMs

## Verify the overlapping from the trace file
* step 1: we need to run two streams and generating the trace.

The following example running 2 streams on dev 0, with data size 1000 floats for each vector.
```bash
nvprof --print-gpu-trace --csv ./mem_mem_ptx_all 2 0 1000 0 2> trace.csv
```
* step 2: use read_trace.ipynb

you can visualize the timeline using the python notebook. Search instances when the two kernels will have execution overlapping.

for mem_mem, 12000 - 300000 seems to be a good overlapping test range.

* step 3: generate the trace for prefound search range

run profile_app.sh in the terminal

It will produce both trace and metrics for all the sampling points.
```bash
./profile_app.sh
```



### Reference
https://github.com/3upperm2n/cke_ovlp_rate
