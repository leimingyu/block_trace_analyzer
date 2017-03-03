* step 1: generate trace and check the read_trace.ipynb
```bash
nvprof --print-gpu-trace --csv ./cmp_cmp_ptx 2 0 40000 0 2> trace.csv
nvprof --print-gpu-trace --csv ./cmp_cmp_ptx 2 0 52000 0 2> trace.csv
```
It seems that from 40000 to 52000, the kernels are overlapping with each other.

* step 2: edit the ./profile_app.sh script and modify the range for benchmarking
```bash
./profile_app.sh
```

* step 3: find out the overlapping ratio for all the test cases and pick 2-3 cases to study.
use search_ovlp_pos.ipynb

