import pandas as pd
import numpy as np
import operator


class transfer():
    def __init__(self, start=0.0,end=0.0):
        self.start_time_ms = start
        self.end_time_ms = end
        
        
class streams():
    def __init__(self):
        self.h2d = []
        self.d2h = []
        self.kernel = []
        
        
def time_coef_ms(df_trace):
    rows, cols = df_trace.shape
    
    start_unit = df_trace['Start'].iloc[0]
    duration_unit = df_trace['Duration'].iloc[0]
    
    start_coef =  1.0
    if start_unit == 's':
        start_coef = 1e3
    if start_unit == 'us':
        start_coef = 1e-3
    
    duration_coef =  1.0
    if duration_unit == 's':
        duration_coef = 1e3
    if duration_unit == 'us':
        duration_coef = 1e-3
        
    return start_coef, duration_coef       
 
    
# read data for the current row
def read_row(df_row, start_coef_ms, duration_coef_ms):
    start_time_ms = float(df_row['Start']) * start_coef_ms
    
    end_time_ms = start_time_ms + float(df_row['Duration']) * duration_coef_ms
    
    stream_id = int(df_row['Stream'])
    
    api_name = df_row['Name'].to_string()
    
    if "DtoH" in api_name:
        api_type = 'd2h'
    elif "HtoD" in api_name:
        api_type = 'h2d'
    else:
        api_type = 'kernel'
    
    return stream_id, api_type, start_time_ms, end_time_ms


def trace2dataframe(trace_file):
    """
    read the trace file into dataframe using pandas
    """
    # There are max 17 columns in the output csv
    col_name = ["Start","Duration","Grid X","Grid Y","Grid Z","Block X","Block Y","Block Z","Registers Per Thread","Static SMem","Dynamic SMem","Size","Throughput","Device","Context","Stream","Name"]

    df_trace = pd.read_csv(trace_file, names=col_name, engine='python')

    rows_to_skip = 0

    # find out the number of rows to skip
    for index, row in df_trace.iterrows():
        if row['Start'] == 'Start':
            rows_to_skip = index
            break
        
    # read the input csv again 
    df_trace = pd.read_csv(trace_file, skiprows=rows_to_skip)
    
    return df_trace


def get_stream_info(df_trace):
    """
    read dataframe into stream list which contains the h2d/d2h/kernel star and end time in ms.
    """
    streamList = []

    # read the number of unique streams
    stream_id_list = df_trace['Stream'].unique()
    stream_id_list = stream_id_list[~np.isnan(stream_id_list)] # remove nan

    num_streams = len(stream_id_list)

    for i in xrange(num_streams):
        streamList.append(streams())
        
    start_coef, duration_coef = time_coef_ms(df_trace)

    # read row by row
    for rowID in xrange(1, df_trace.shape[0]):
        #  extract info from the current row
        stream_id, api_type, start_time_ms, end_time_ms = read_row(df_trace.iloc[[rowID]], start_coef, duration_coef)

        # find out index of the stream 
        sid, = np.where(stream_id_list==stream_id)

        # add the start/end time for different api calls
        if api_type == 'h2d':
            streamList[sid].h2d.append(transfer(start_time_ms, end_time_ms))
        elif api_type == 'd2h':
            streamList[sid].d2h.append(transfer(start_time_ms, end_time_ms))
        elif api_type == 'kernel':
            streamList[sid].kernel.append(transfer(start_time_ms, end_time_ms))
        else:
            print "Unknown. Error."
            
    return streamList


def check_kernel_ovlprate(trace_file):
    """
    Read the trace file and figure out the overlapping rate for the two kernel execution.
    """
    # read data from the trace file
    df_trace = trace2dataframe(trace_file)
    
    # extract stream info
    streamList = get_stream_info(df_trace)
    
    # check kernel overlapping
    preK_start = streamList[0].kernel[0].start_time_ms
    preK_end = streamList[0].kernel[0].end_time_ms

    curK_start = streamList[1].kernel[0].start_time_ms
    curK_end = streamList[1].kernel[0].end_time_ms

    preK_runtime = preK_end - preK_start
    curK_runtime = curK_end - curK_start

    ovlp_duration = preK_end - curK_start
    ovlp_ratio = ovlp_duration / preK_runtime

#    if curK_start >= preK_start and curK_start <= preK_end:
#        print('concurrent kernel execution :\n\t stream-prev {} ms \n\t stream-cur {} ms'
#        '\n\t overlapping {} ms \n\t ovlp ratio (based on prev stream) {}%'\
#              .format(preK_runtime, curK_runtime, ovlp_duration, ovlp_ratio))

    return ovlp_ratio


def get_kernel_time_from_trace(df_trace):
    """
    Read kernel time from trace.
    """
    # read the number of unique streams
    stream_id_list = df_trace['Stream'].unique()
    stream_id_list = stream_id_list[~np.isnan(stream_id_list)] # remove nan

    start_coef, duration_coef = time_coef_ms(df_trace)

    kernel_time_dd = {}

    # read row by row
    for rowID in xrange(1, df_trace.shape[0]):
        #  extract info from the current row
        stream_id, api_type, start_time_ms, end_time_ms = \
        read_row(df_trace.iloc[[rowID]], start_coef, duration_coef)

        # find out index of the stream
        sid, = np.where(stream_id_list == stream_id)

        sid = int(sid)
        # find out the duration for kernel
        if api_type == 'kernel':
            duration = end_time_ms - start_time_ms
            kernel_time_dd[sid] = duration

    return kernel_time_dd


def kernel_slowdown(s1_kernel_dd, s2_kernel_dd):
    slow_down_ratio_list = []
    for key, value in s2_kernel_dd.items():
        v_s1 = s1_kernel_dd[0]
        slow_down_ratio_list.append(value / float(v_s1))
    return slow_down_ratio_list
