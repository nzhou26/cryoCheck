#!/usr/bin/env python3
import sys
from mrc_to_png import toPNG
from cryocheck_infer_png import pred

# %%
if __name__ == '__main__':
    toPNG(sys.argv[1])
    png_dir = f'{sys.argv[1]}/png/'
    pred(png_dir)
    



