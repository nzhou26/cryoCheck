# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import random
from importlib_metadata import pathlib
import os
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import *
import pandas as pd


# %%
IMG_SIZE = (320,320)
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 32
model_save_dir = '/storage_data/zhou_Ningkun/cryocheck_data_model/models'
base_models = [
    # 'DenseNet121',
    # 'DenseNet169',
    # 'DenseNet201',
    # 'EfficientNetB0',
    # 'EfficientNetB1',
    # 'EfficientNetB2',
    # 'EfficientNetB3',
    # 'EfficientNetB5',
    'EfficientNetB6',
    'EfficientNetB7',
    'InceptionResNetV2',
    'InceptionV3',
    'MobileNet',
    'MobileNetV2',
    'MobileNetV3Large',
    'MobileNetV3Small',
    'ResNet101',
    'ResNet101V2',
    'ResNet152',
    'ResNet152V2',
    'ResNet50',
    'ResNet50V2',
    'VGG16',
    'VGG19',
    'Xception'
]
def train_current(dataset, base_model_name, fine_tune_percent=0.2, gpu_list=[0,1]):
    train_set, val_set, test_set, num_used = dataset
    gpu_use = [f'/gpu:{gpu}' for gpu in gpu_list]
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=gpu_use)
    with mirrored_strategy.scope():
        class_method = getattr(tf.keras.applications, base_model_name)
        base_model = class_method(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        model = create_model(base_model)
        base_learning_rate = 0.001
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                    loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=[tf.keras.losses.BinaryCrossentropy(from_logits=True, 
                                                                name='binary_crossentropy'),
                                                                f1_metric,
                                                                'accuracy'])
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=5)
    initial_epochs = 30
    print(f'{base_model_name} train...')
    history = model.fit(train_set, 
                    epochs=30, 
                    validation_data=val_set,
                    callbacks=[es_callback],
                    verbose=0)
    print('evaluate...')
    loss_test, binary_cross_entropy_test , f1_score_test,accuracy_test = model.evaluate(test_set)
    
    hist_df = pd.DataFrame(history.history)
    accuracy_test = "{0:.2%}".format(accuracy_test)[:-1]
    f1_score_test = "{0:.2%}".format(f1_score_test)[:-1]
    result_base_name = f'acc-{accuracy_test}--f1-{f1_score_test}--len-{num_used}--{base_model_name}'
    hist_csv = f'{model_save_dir}/{result_base_name}--history.csv'
    model_file_name = f'{model_save_dir}/{result_base_name}--model.h5'
    with open(hist_csv, mode='w') as f:
        hist_df.to_csv(f)
    model.save(f'{model_file_name}', overwrite=True)
    
    # fine tuning
    fine_tune_at = round(len(base_model.layers)*(1- fine_tune_percent))
    with mirrored_strategy.scope():
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    optimizer= tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
                    metrics=[tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'), 
                    f1_metric, 'accuracy'])
    fine_tune_epochs = 20
    total_epochs = initial_epochs + fine_tune_epochs
    print('fine_tune...')

    history_fine = model.fit(train_set,
                            epochs=total_epochs,
                            initial_epoch=history.epoch[-1],
                            validation_data=val_set,
                            callbacks=[es_callback],
                            verbose=0)
    print('evaluate...')
    loss_test, binary_cross_entropy_test , f1_score_test,accuracy_test = model.evaluate(test_set)
    hist_df = pd.DataFrame(history_fine.history)
    accuracy_test = "{0:.2%}".format(accuracy_test)[:-1]
    f1_score_test = "{0:.2%}".format(f1_score_test)[:-1]
    result_base_name = f'acc-{accuracy_test}--f1-{f1_score_test}--len-{num_used}--{base_model_name}-ft'
    hist_csv = f'{model_save_dir}/{result_base_name}--history.csv'
    model_file_name = f'{model_save_dir}/{result_base_name}--model.h5'
    with open(hist_csv, mode='w') as f:
        hist_df.to_csv(f)
    model.save(f'{model_file_name}', overwrite=True)

def test_acc_diff_model():
    return None
def create_dataset(num_to_train):
    data_dir = '/ssd/train_data_cryocheck'
    os.system('rm -rf /ssd/tmp_cryocheck')
    os.system('mkdir -p /ssd/tmp_cryocheck/good')
    os.system('mkdir -p /ssd/tmp_cryocheck/bad')
    good_paths = list(pathlib.Path(data_dir).glob('*/good/*.png'))
    bad_paths = list(pathlib.Path(data_dir).glob('*/bad/*.png'))
    good_ratio = len(good_paths)/(len(good_paths) + len(bad_paths))
    bad_ratio = len(bad_paths)/(len(good_paths) + len(bad_paths))
    random.seed(222)
    random.shuffle(good_paths)
    random.shuffle(bad_paths)
    if num_to_train != -1:
        # slice training set if use a small portion of data
        good_paths = good_paths[:int(num_to_train*good_ratio)]
        bad_paths = bad_paths[:int(num_to_train*bad_ratio)] 
    print('creating soft links')
    for item in good_paths:
        os.system(f'ln -snf {item.resolve()} /ssd/tmp_cryocheck/good/')
    for item in bad_paths:
        os.system(f'ln -snf {item.resolve()} /ssd/tmp_cryocheck/bad/')
    PATH = os.path.join('/ssd/tmp_cryocheck')
    
    
    train_dataset = image_dataset_from_directory(PATH,
                                                validation_split=0.2,
                                                subset = 'training',
                                                #color_mode='grayscale',
                                                seed = 133,
                                                image_size=IMG_SIZE,
                                                batch_size = BATCH_SIZE)
    validation_dataset = image_dataset_from_directory(PATH,
                                                    validation_split=0.2,
                                                    subset='validation',
                                                    #color_mode='grayscale',
                                                    seed=133,
                                                    image_size=IMG_SIZE,
                                                    batch_size=BATCH_SIZE)

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches//5)
    validation_dataset = validation_dataset.skip(val_batches //5)
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    return (train_dataset, validation_dataset, test_dataset, len(good_paths) + len(bad_paths))
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# %%
def create_model (base_model):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(40),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0,0.2),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2,0)
    ])
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model


# %%
if __name__ == '__main__':
    
    dataset = create_dataset(-1)

    for model in base_models:
        train_current(dataset, 
        base_model_name=model, 
        gpu_list=[1,2,3]
        )

    

