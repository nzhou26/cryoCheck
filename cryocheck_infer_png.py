#!/usr/bin/env python3
import pathlib
import os
import tensorflow as tf
import numpy as np
import sys

model_path = '/storage_data/zhou_Ningkun/cryocheck/saved_model/efficientnetb0_not_augmented_0.9_2021-11-10_ft.h5'
model = tf.keras.models.load_model(model_path, compile=False)

# %%
png_mrc_dir = pathlib.Path(sys.argv[1])
png_mrc_paths = list(png_mrc_dir.glob('*.png'))
preds =[]
os.system(f'mkdir -p {sys.argv[1]}/good_img')
os.system(f'mkdir -p {sys.argv[1]}/bad_img')
#mrc_dir = os.path.abspath('preprocessing/MotionCorr/job002/movies')
for png_mrc_path in png_mrc_paths:
    try:
        img = tf.keras.preprocessing.image.load_img(png_mrc_path, target_size=model.layers[0].input_shape[0][1:3])
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        pred = model.predict(images, batch_size=10)
        pred = tf.nn.sigmoid(pred)
        pred = tf.where(pred < 0.5, 0, 1)
        pred = bool(pred == 1)
        if pred:
            print(f"good {png_mrc_path}")
            good_path = png_mrc_path.name[:-4]
            os.system(f'ln -snf {png_mrc_path.resolve()} {sys.argv[1]}/good_img/')
            preds.append(pred)
        else:
            print(f"bad {png_mrc_path}")
            os.system(f'ln -snf {png_mrc_path.resolve()} {sys.argv[1]}/bad_img/')
    except KeyboardInterrupt:
        exit()

print(f'Done! It selected {len(preds)} good images from {len(png_mrc_paths)} images')

# %%



