### Author
Leiming Yu
* Email: ylm@ece.neu.edu
* Twitter: @yu_leiming
* Blog: http://www1.coe.neu.edu/~ylm/

### Motivation
Analyze the concurrent kernel execution.

### Folders
* mem_mem : memory bound kernel 
* cmp_cmp : compute bound kernel

### Techniques
* inline ptx for thread timing and other trace info
* using unified memory to store the trace
* when dumpying the trace info (for all the gpu threads), the clock() is normalized by 1e-6.


### Examples
The commandline option 

./mem_mem_ptx_all stream_num device_id data_size print_log

An examples,

./mem_mem_ptx_all 1 0 1000 0




### Reference
https://github.com/3upperm2n/cke_ovlp_rate
