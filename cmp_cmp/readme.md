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
```bash
overlapping rate : 0.295503908039 data size : 43600
overlapping rate : 0.205063858391 data size : 50200
overlapping rate : 0.115285117999 data size : 47000
```

* step 4: use **2stream_sm_blk_plot.ipynb** to plot the block timeline for each SM on your device.

generate the trace info (turn on 1) for all the threads and dump the data in csv file.
```bash
./cmp_cmp_ptx 2 0 47000 1 > tmp_47000.csv
./cmp_cmp_ptx 2 0 50200 1 > tmp_50200.csv
./cmp_cmp_ptx 2 0 43600 1 > tmp_43600.csv
```

use the notebook to plot the trace.
