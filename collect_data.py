#!/usr/bin/env python3
import sys
import pathlib
import os
from difflib import SequenceMatcher
def collect_png(png_dir, goodtif_dir, dest_dir):
    data_name = os.path.basename(os.path.dirname(goodtif_dir))
    os.system(f'mkdir -p {dest_dir}/{data_name}')
    os.system(f'mkdir -p {dest_dir}/{data_name}/good')
    os.system(f'mkdir -p {dest_dir}/{data_name}/bad')
    correct_len = len(list(pathlib.Path(goodtif_dir).glob('*.tif')))
    good_png_list = []
    png_list = list(pathlib.Path(png_dir).glob('*.png'))

    for tif in pathlib.Path(goodtif_dir).glob('*.tif'):
        found = False
        for png in png_list:
            if SequenceMatcher(None,png.name[:-8],tif.name[:-4]).ratio() > 0.9:
                good_png_list.append(png)
                png_list.remove(png)
                found = True
                print(f'{len(png_list)} png left to be classified')
                break
        if not found:
            print(f'png not found: {tif}')
    print(len(good_png_list))
    print(correct_len)
    if len(good_png_list) == correct_len:
        for png in good_png_list:
            os.system(f'mv -v {png} {dest_dir}/{data_name}/good')
        os.system(f'mv -v {png_dir}/*.png {dest_dir}/{data_name}/bad')
    else:
        print('pattern in tif does not match pattern in png')
    
if __name__ == "__main__":
    if sys.argv[1] == '-h':
        print('usage: collect_data.py png_dir goodtif_dir dest_dir')
        exit()
    collect_png(png_dir= sys.argv[1], goodtif_dir=sys.argv[2], dest_dir=sys.argv[3])
    