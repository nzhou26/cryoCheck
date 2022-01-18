# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory
from datetime import date
from tensorflow.keras.applications import *
import pandas as pd


# %%

percent_layers_to_use = 0.2

os.system('mkdir -p train_result')
os.system('mkdir -p train_result/saved_models')
os.system('mkdir -p train_result/figures')

PATH = os.path.join('current_train')
BATCH_SIZE = 32
IMG_SIZE = (300,300)
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
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# %%
def create_model (train_dataset, IMG_SHAPE, base_model, preprocess_input, augmented):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(40),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0,0.2),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2,0)
    ])
    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    if augmented: 
        x = data_augmentation(inputs)
        x = preprocess_input(x)
    else:
        x= preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model


# %%
def train_model(train_dataset, validation_dataset, test_dataset, IMG_SHAPE, base_model, 
                preprocess_input, augmented, learning_rate, percent_layers_to_use):
    if augmented:
        standard_name = base_model.name + '_augmented_'
    else:
        standard_name = base_model.name + '_not_augmented_'
    model = create_model(train_dataset, IMG_SHAPE, base_model, preprocess_input, augmented=augmented)
    base_learning_rate = learning_rate
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.losses.BinaryCrossentropy(from_logits=True, 
                                                            name='binary_crossentropy'),
                                                            f1_metric,
                                                            'accuracy'])
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=3)
    model.summary()
    initial_epochs = 30
    loss0, _, _, accuracy0 = model.evaluate(validation_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))
    history = model.fit(train_dataset, 
                    epochs=initial_epochs, 
                    validation_data=validation_dataset,
                    callbacks=[es_callback],
                    verbose=0)
    loss_test, binary_cross_entropy_test , f1_score_test,accuracy_test = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy_test)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    #plt.savefig('pretrained_result/figures/' + standard_name + '_pre_ft.png')
    if augmented:
        model_file = 'train_result/saved_models/' +                      standard_name +                      str(round(accuracy_test,2)) + '_' + str(date.today()) + '_pre_ft.h5'
    else: 
        model_file = 'train_result/saved_models/' +                      standard_name +                       str(round(accuracy_test,2)) + '_' + str(date.today()) + '_pre_ft.h5'
    model.save(model_file, overwrite=True)
    base_model.trainable = True
    print('Number of layers in the base model: ', len(base_model.layers))
    fine_tune_at = round(len(base_model.layers)*(1- percent_layers_to_use))
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer= tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
                  metrics=[tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'), 
                  f1_metric, 'accuracy'])
    model.summary()
    fine_tune_epochs = 20
    total_epochs = initial_epochs + fine_tune_epochs
    history_fine = model.fit(train_dataset,
                            epochs=total_epochs,
                            initial_epoch=history.epoch[-1],
                            validation_data=validation_dataset,
                            callbacks=[es_callback],
                            verbose=0)
    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([max(history.epoch) -1,max(history.epoch ) - 1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([max(history.epoch) -1,max(history.epoch ) - 1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('train_result/figures/' + standard_name + '_ft.png')
    loss_test_ft, binary_cross_entropy_test_ft , f1_score_test_ft,accuracy_test_ft = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy_test_ft)
    if augmented:
        model_file_tf = 'train_result/saved_models/' +                      standard_name +                      str(round(accuracy_test_ft,2)) + '_' + str(date.today()) + '_ft.h5'
    else: 
        model_file_tf = 'train_result/saved_models/' +                      standard_name +                      str(round(accuracy_test_ft,2)) + '_' + str(date.today()) + '_ft.h5'
    model.save(model_file_tf, overwrite=True)
    return {'f1_score_test': f1_score_test, 
            'accuracy_test': accuracy_test, 
            'number_of_trained_variables': len(model.trainable_variables), 
            'f1_score_test_ft': f1_score_test_ft, 
            'accuracy_test_ft': accuracy_test_ft}


# %%
IMG_SHAPE = IMG_SIZE + (3,)
base_models = [mobilenet_v2.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights=None),
               MobileNetV3Small(input_shape=IMG_SHAPE, include_top=False, weights=None),
              resnet50.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights=None),
              resnet_v2.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights=None),
              vgg16.VGG16(input_shape=IMG_SHAPE, include_top=False, weights=None),
              inception_v3.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights=None),
              efficientnet.EfficientNetB0(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')]
base_models[0].load_weights("pre-trained_weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5")
base_models[1].load_weights("pre-trained_weights/weights_mobilenet_v3_small_224_1.0_float_no_top.h5")
base_models[2].load_weights("pre-trained_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
base_models[3].load_weights("pre-trained_weights/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5")
base_models[4].load_weights("pre-trained_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
base_models[5].load_weights("pre-trained_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")

preprocess_inputs = [mobilenet_v2.preprocess_input,
                     mobilenet_v3.preprocess_input,
                     resnet50.preprocess_input,
                     resnet_v2.preprocess_input,
                     vgg16.preprocess_input,
                     inception_v3.preprocess_input,
                     efficientnet.preprocess_input]


# %%
results = {}
for base_model, preprocess_input in zip(base_models, preprocess_inputs):
  print(base_model.name)
  results[base_model.name] = train_model(train_dataset, validation_dataset,test_dataset, 
                                                        IMG_SHAPE, base_model,preprocess_input, 
                                                        augmented=False, learning_rate = 0.001, 
                                                        percent_layers_to_use=percent_layers_to_use)
  results[base_model.name + '_augmented'] = train_model(train_dataset, validation_dataset,test_dataset, 
                                                                       IMG_SHAPE, base_model,preprocess_input, 
                                                                       augmented=True, learning_rate = 0.001, 
                                                                       percent_layers_to_use=percent_layers_to_use)


# %%
df = pd.DataFrame.from_dict(results)


# %%
df.to_csv('train_result/bulk_pre_trained.csv')


# %%



