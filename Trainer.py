import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_datasets(working_dir, batch_size, image_size):

    image_size = (image_size, image_size)
    train_dir = os.path.join(working_dir, 'train')
    validation_dir = os.path.join(working_dir, 'validation')
    test_dir = os.path.join(working_dir, 'test')

    datagen = ImageDataGenerator()
    train_dataset = datagen.flow_from_directory(train_dir,
                                            target_size=image_size,
                                            batch_size=batch_size)

    validation_dataset = datagen.flow_from_directory(validation_dir,
                                            target_size=image_size,
                                            batch_size=batch_size)

    test_dataset = datagen.flow_from_directory(test_dir,
                                            target_size=image_size,
                                            batch_size=batch_size)

    datasets = {'train':train_dataset,'test':test_dataset,'validation':validation_dataset}
    return datasets

def get_data_augmentations(augmentations):

    layers = []
    for augment in augmentations:
        if augment == 'random_flip':
            layers.append(tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'))
        elif augment == 'random_rotation':
            layers.append(tf.keras.layers.experimental.preprocessing.RandomRotation(0.2))
    
    data_augmentation = tf.keras.Sequential(layers)
    return data_augmentation

def get_model(model_name, n_classes, image_size, augmentations, base_learning_rate):

    image_size = (image_size, image_size)
    image_shape = image_size + (3,)
    if model_name == 'mobilenet_v2':
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                                        include_top=False,
                                                        weights='imagenet')
    else:
        raise Exception('invalid argument: model not supported')
    
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(n_classes)
    data_augmentation = get_data_augmentations(augmentations)

    inputs = tf.keras.Input(shape=image_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def train_model(model, train_dataset, validation_dataset, epochs):

    history = model.fit(
        train_dataset, 
        epochs=epochs,
        validation_data=validation_dataset
    )

    return model, history

def get_training_report(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)
    report = pd.DataFrame({
        "epoch"                 :   epochs,
        "accuracy"              :   acc,
        "valdidation accuracy"  :   val_acc,
        "loss"                  :   loss,
        "validation loss"       :   val_loss
    })
    return report