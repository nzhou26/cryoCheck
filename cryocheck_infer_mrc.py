#!/usr/bin/env python3
import sys
import os
from importlib_metadata import pathlib
from mrc_to_png import toPNG
from cryocheck_infer_png import pred

def select_from_dir(png_dir, mrc_dir):
    good_png_paths = list(pathlib.Path(png_dir).glob('good_img/*.png'))
    good_png_names = [item.name[:-4] for item in good_png_paths]
    os.system('mkdir -p good_mrc')
    for name in good_png_names:
        mrc_path = os.path.abspath(f'{mrc_dir}/{name}')
        os.system(f'ln -snf {mrc_path} good_mrc/')
# %%
if __name__ == '__main__':
    mrc_dir = sys.argv[1]
    toPNG(sys.argv[1])
    png_dir = f'{mrc_dir}/png/'
    pred(png_dir)
    select_from_dir(png_dir, mrc_dir)
    print('Done! good mrc sym links stored in good_mrc/')
    



