#!/usr/bin/env python3
import pathlib
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import sys

# %%
# model_dir = pathlib.Path('/storage_data/zhou_Ningkun/cryocheck/saved_model')
# list_of_models = list(model_dir.glob('*.h5'))
# def acc(file_name):
#     return(float(file_name.name.split('_')[1]))
# model_path = max(list_of_models, key=acc)
model_path = '/storage_data/zhou_Ningkun/cryocheck/saved_model/efficientnetb0_not_augmented_0.9_2021-11-10_ft.h5'
model = tf.keras.models.load_model(model_path, compile=False)

# %%
os.system(f'/storage_data/zhou_Ningkun/cryocheck/mrc_to_png_on_the_fly.py {sys.argv[1]} 341 480')
png_mrc_dir = pathlib.Path('png_mrc')
png_mrc_paths = list(png_mrc_dir.glob('*.png'))
df = pd.DataFrame(png_mrc_paths)
preds =[]
for png_mrc_path in png_mrc_paths:
    #print(png_mrc_path)
    try:
        img = tf.keras.preprocessing.image.load_img(png_mrc_path, target_size=model.layers[0].input_shape[0][1:3])
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        pred = model.predict(images, batch_size=10)
        pred = tf.nn.sigmoid(pred)
        pred = tf.where(pred < 0.5, 0, 1)
        pred = bool(pred == 1)
        preds.append(pred)
    except:
        print(png_mrc_path)
df['pred'] = preds

spliter = list(df[df['pred']][0])
os.system('mkdir -p goodmrc_auto')

mrc_dir = os.path.abspath(sys.argv[1])
for item in spliter:
    good = str(item).split('/')[-1][:-4]
    os.system(f'ln -snf {mrc_dir}/{good} goodmrc_auto/')
print(f'Done! It picked {len(spliter)} good images from {len(df)} images')

# %%



