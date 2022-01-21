#!/usr/bin/env python3
import pathlib
import os
import tensorflow as tf
import numpy as np
import sys

# edit model path here to the model file you downloaded from google drive
MODEL_PATH = '/storage_data/zhou_Ningkun/cryocheck_data_model/models/acc-88.32--f1-91.90--len-70941--ResNet101-ft--model.h5'
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# %%
def pred(png_dir):
    png_mrc_dir = pathlib.Path(png_dir)
    png_mrc_paths = list(png_mrc_dir.glob('*.png'))
    preds =[]
    os.system(f'mkdir -p {png_dir}/good_img')
    os.system(f'mkdir -p {png_dir}/bad_img')
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
                os.system(f'ln -snf {png_mrc_path.resolve()} {png_dir}/good_img/')
                preds.append(pred)
            else:
                print(f"bad {png_mrc_path}")
                os.system(f'ln -snf {png_mrc_path.resolve()} {png_dir}/bad_img/')
        except KeyboardInterrupt:
            exit()

    print(f'Done! It selected {len(preds)} good images from {len(png_mrc_paths)} images')

if __name__ =='__main__':
    pred(sys.argv[1])