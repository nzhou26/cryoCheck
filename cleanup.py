#!/usr/bin/env python3
'''
This script recognise sym links in good mrc directory and replace them with true file 
and select out good tif based on pattern in good mrc directory.
'''
import pathlib
import os
import sys
from difflib import SequenceMatcher
def clean_up(good_mrc_dir, total_tif_dir):
    good_tif_paths = []
    good_mrc_paths = []
    for good_mrc in  pathlib.Path(good_mrc_dir).glob('*.mrc'):
        good_mrc_name = good_mrc.name[:-4]
        good_mrc_paths.append(os.readlink(good_mrc))
        for tif in pathlib.Path(total_tif_dir).glob('*.tif'):
            if SequenceMatcher(None,good_mrc_name,tif.name[:-4]).ratio() > 0.9:
                good_tif_paths.append(tif)
    
    if len(good_mrc_paths) == len(good_tif_paths):
        for mrc,tif in zip(good_mrc_paths, good_tif_paths):
            os.system(f'mv -v {mrc} good_tif_mrc/mrc')
            os.system(f'mv -v {tif} good_tif_mrc/tif')

if __name__ == '__main__':
    print('This step will move your tif and mrc files into a new folder called good_tif_mrc/, please make sure you files are stored in the correct locations.')
    print('THIS STEP CANNOT BE UNDONE!!!')
    inp = input('yes or no: ')
    if inp != 'yes':
        print('exit')
        exit()
    os.system('mkdir -p good_tif_mrc')
    os.system('mkdir -p good_tif_mrc/tif')
    os.system('mkdir -p good_tif_mrc/mrc')
    clean_up(sys.argv[1], sys.argv[2])
