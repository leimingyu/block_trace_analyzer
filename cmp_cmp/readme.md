* step 1: generate trace and check the read_trace.ipynb
```bash
nvprof --print-gpu-trace --csv ./cmp_cmp_ptx 2 0 40000 0 2> trace.csv
nvprof --print-gpu-trace --csv ./cmp_cmp_ptx 2 0 52000 0 2> trace.csv
```
It seems that from 40000 to 52000, the kernels are overlapping with each other.
