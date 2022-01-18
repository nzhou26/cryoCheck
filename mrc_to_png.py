#!/usr/bin/env python
import mrcfile
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pathlib
import multiprocessing
from datetime import datetime
import random

process_number = 8

binX = 341
binY = 480
class myProcess (multiprocessing.Process):
    def __init__(self, processID, name, data_paths, start_idx, end_idx, output):
        multiprocessing.Process.__init__(self)
        self.processID = processID
        self.name = name
        self.data_paths = data_paths
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.output = output
    def run(self):
        convert(self.name, self.data_paths, self.start_idx, self.end_idx, self.output)
def convert(processName, data_paths, idx, end_idx, output):
    while idx < end_idx:
        if processName == "Process-0":
                percent = "{:.2%}".format(idx/end_idx)
                output_message = f'{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}   Preprocessing, progress: {percent}'            
                print(output_message)
        try:
            item = data_paths[idx]
            if not os.path.isfile(f'{output}' + item.name + '.png'):
                mrc_arr = read_mrc(item)
                arr_01 = (mrc_arr - np.min(mrc_arr))/np.ptp(mrc_arr)
                plt.imsave(f'{output}' + item.name + '.png' , arr_01, cmap='gray')
                #print('converting ' + item.name +' to png')
        except KeyboardInterrupt:
            print('keyboard catched')
            break
        idx += 1

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)
def read_mrc(file_name):
    with mrcfile.open(file_name) as mrc:
        arr = mrc.data
        #print(arr.shape)
        arr = rebin(arr, [binX, binY])
    return arr
def toPNG(mrc_dir):
    data_dir = mrc_dir
    data_dir = pathlib.Path(data_dir)
    data_paths  = list(data_dir.glob('*.mrc'))
    items_to_be_done = len(data_paths)
    items_for_each_process = items_to_be_done // process_number
    random.shuffle(data_paths)
    processes = []
    output_dir = f'{mrc_dir}/png/'
    os.system(f'mkdir -p {mrc_dir}/png')

    for i in range(process_number):
        start_idx = i*items_for_each_process
        end_idx = start_idx+items_for_each_process
        if i == process_number -1 :
            end_idx = items_to_be_done
        processes.append( myProcess(i, f"Process-{i}", data_paths, start_idx, end_idx, output_dir))

    for process in processes:
        process.start()
    for process in processes:
        process.join()
if __name__ == '__main__':
    toPNG(sys.argv[1])