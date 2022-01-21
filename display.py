# %%
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import pathlib
import random
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from train import f1_metric
import tensorflow as tf
import os
IMG_SIZE = (320,320)
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 32
data_dir = '/storage_data/zhou_Ningkun/cryocheck/train_data'
model_dir = '/storage_data/zhou_Ningkun/cryocheck_data_model/models'
def show_manual(num_to_display):
    img_paths = list(pathlib.Path(data_dir).glob('*/*/*.png'))
    random.shuffle(img_paths)
    batch = img_paths[:num_to_display]
    width =  round(np.sqrt(num_to_display))
    plt.figure(figsize=(10,10))
    for i in range(1, width*width +1):
        img = plt.imread(batch[i-1])
        title = batch[i-1].parent.name
        plt.subplot(width, width, i)
        plt.imshow(img)
        plt.axis('off')
        if title == 'bad':
            plt.title(title, color='red')
        else:
            plt.title(title, color='green')
    plt.tight_layout()
    plt.show()
def rank_models(model_keyword):
    model_paths = list(pathlib.Path(model_dir).glob(model_keyword))
    model_names = [model.name for model in model_paths]
    model_acc = [name.split('--')[0].split('-')[1] for name in model_names]
    model_f1 = [name.split('--')[1].split('-')[1] for name in model_names]
    model_base = [name.split('--')[3] for name in model_names]
    df = pd.DataFrame({
        'Base Model': model_base,
        'Accuracy':model_acc,
        'F1 Score': model_f1
        
    })
    df = df.sort_values(by='Accuracy',ascending=False)
    display(df[:10])
def stats(data_dir, model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=[tf.keras.losses.BinaryCrossentropy(from_logits=True, 
                                                                name='binary_crossentropy'),
                                                                f1_metric,
                                                                'accuracy'])
    data_names = []
    f1s = []
    accs = []
    for dir in pathlib.Path(data_dir).glob('*'):
        data_names.append(dir.name)
        print(f'Testing: {dir.name}')
        f1, acc =infer_dir(dir, model)
        f1s.append(f1)
        accs.append(acc)
    
    df = pd.DataFrame({
        'dataset name': data_names,
        'accuracy': accs,
        'F1 score': f1s
    })
    df = df.sort_values(by='accuracy',ascending=False)
    display(df)
    df.to_csv(f'/storage_data/zhou_Ningkun/cryocheck_data_model/stats/{os.path.basename(model_path)}.csv')
def infer_dir(dir, model):
    test_set = image_dataset_from_directory(dir,
                                                validation_split=0.2,
                                                subset='validation',
                                                #color_mode='grayscale',
                                                seed=133,
                                                image_size=IMG_SIZE,
                                                batch_size=BATCH_SIZE)
    loss, bce, f1, acc = model.evaluate(test_set)
    return f1, acc
def stats_hist(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
    plt.xlim(0.65,1)
    plt.xlabel('Accuracy distribution')
    acc_list = df['accuracy'].astype(float).values.tolist()

    n, bins, patches = plt.hist(acc_list, 20, facecolor='springgreen', alpha=0.7)
    print('mean')
    print(np.average(acc_list))
    print('median')
    print(np.median(acc_list))
    print('standard deviation')
    print(np.std(acc_list))
if __name__ == '__main__':
    #show_manual(9)
    #rank_models('*len-7*.h5')

    model_path = '/storage_data/zhou_Ningkun/cryocheck_data_model/models/acc-88.32--f1-91.90--len-70941--ResNet101-ft--model.h5'
    data_dir = '/ssd/train_data_cryocheck/'
    #stats(data_dir, model_path)
    
    csv_file = '/storage_data/zhou_Ningkun/cryocheck_data_model/stats/acc-88.32--f1-91.90--len-70941--ResNet101-ft--model.h5.csv'
    stats_hist(csv_file)

# %%
