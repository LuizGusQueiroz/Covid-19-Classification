import tensorflow as tf

import opendatasets as op
op.download("https://www.kaggle.com/tawsifurrahman/covid19-radiography-database")

op.download("https://www.kaggle.com/datasets/sinamhd9/chexnet-weights")

import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
import pathlib
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC

dataset_folder = os.path.join("covid19-radiography-database/COVID-19_Radiography_Dataset")

files_not_important = ["COVID.metadata.xlsx",
                       "Lung_Opacity.metadata.xlsx",
                       "Normal.metadata.xlsx",
                       "README.md.txt",
                       "Viral Pneumonia.metadata.xlsx"]
for i in files_not_important:
    os.remove(os.path.join(dataset_folder, i))


import shutil
files_not_important = [
                       "COVID/masks",
                       "Lung_Opacity",
                       "Normal",
                       "Viral Pneumonia/masks"]
for i in files_not_important:
  shutil.rmtree(os.path.join(dataset_folder, i), ignore_errors=True)

datasetObject = pathlib.Path(os.path.join(dataset_folder))
images = list(datasetObject.glob("*/*/*.*"))

len(images)

image_data_generator = ImageDataGenerator(
    rescale=1 / 255, vertical_flip=False, horizontal_flip=True, zoom_range=0.1, zca_whitening=False,
    samplewise_center=True, samplewise_std_normalization=True, validation_split=0.1,
    rotation_range=0.2)
training_dataset = image_data_generator.flow_from_directory(
    dataset_folder, target_size=(224, 224), color_mode='rgb', subset='training', batch_size=8, shuffle=True
)
validation_dataset = image_data_generator.flow_from_directory(
    dataset_folder, target_size=(224, 224), color_mode='rgb', subset='validation', batch_size=8, shuffle=True
)

single_batch = training_dataset.next()
images = single_batch[0]
label = single_batch[1]
plt.figure(figsize=(20, 10))
for i in range(8):
    plt.subplot(2, 4, (i + 1))
    plt.imshow(images[i])
    plt.title(label[i])
plt.show()

training_dataset.classes

training_dataset.class_indices

np.asarray(images[0]).shape


np.unique(images[0])

from keras.applications import densenet
from keras.initializers import GlorotNormal
d = densenet.DenseNet121(weights=None, include_top = False, input_shape = (224, 224, 3))

print(d.output_shape)
m = tf.keras.layers.Dropout(0.7)(d.output)
m = tf.keras.layers.GlobalAveragePooling2D()(m)
m = tf.keras.layers.Dropout(0.7)(m)
m = tf.keras.layers.Dense(2, kernel_initializer=GlorotNormal(),
                          activation='softmax', kernel_regularizer=tf.keras.regularizers.L2(0.0001),
                          bias_regularizer=tf.keras.regularizers.L2(0.0001))(m)
m = tf.keras.models.Model(inputs=d.input, outputs=m)
m.load_weights("chexnet-weights/brucechou1983_CheXNet_Keras_0.3.0_weights.h5", by_name=True, skip_mismatch=True)
for layer in m.layers[:200]:
    layer.trainable = False
for layer in m.layers[200:]:
    layer.trainable = True

m.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
          , loss = 'categorical_crossentropy', metrics =  [TruePositives(name='tp'),
                                                          FalsePositives(name='fp'),
                                                          TrueNegatives(name='tn'),
                                                          FalseNegatives(name='fn'),
                                                          'accuracy',
                                                          Precision(name='precision'),
                                                          Recall(name='recall')])

m.summary()

ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, mode='min', patience=2)

history = m.fit(
    training_dataset,
    validation_data=validation_dataset,
    batch_size=8,
    epochs=26,
    callbacks=[ReduceLROnPlateau,
               tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='min',
                                                restore_best_weights=True)]
)

plt.figure(figsize = (10, 5))
plt.plot(history.history['accuracy'], label="accuracy")
plt.plot(history.history['val_accuracy'], label="val_accuracy")
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.legend()

m.evaluate(validation_dataset, batch_size=8)

single_image = cv2.imread(
    os.path.join(dataset_folder, "COVID", "images", "COVID-163.png")
)
single_image = cv2.resize(single_image, (224, 224))
single_image = cv2.cvtColor(single_image, cv2.COLOR_BGR2RGB)

plt.imshow(single_image)
plt.show()

single_image.shape

img = single_image / 255

img = np.asarray(img)
img = img.reshape(1, 224, 224, 3)
img.shape

y_pred = m.predict(img)[0]

y_pred

y_pred = np.argmax(y_pred)

y_pred


classes = list({'COVID': 0, 'Viral Pneumonia': 1})

classes[y_pred]

# Retrieve the weights from the last layer, without paying attention to the bias values
weights = m.layers[-1].get_weights()[0]

weights

weights = np.asarray(weights)

# we have two class so we have two array in each one we have 1024 weights
weights.shape

weights = weights.reshape(weights.shape[1], weights.shape[0])

weights.shape

# get weights for the predication class
weights_for_predicted_class_for_this_image = weights[y_pred]

weights_for_predicted_class_for_this_image

# get out of last convolution layer by name
new_model = tf.keras.models.Model(
    m.input,
    m.get_layer('conv5_block16_concat').output
)

output_con_layer = new_model.predict(img)[0]

output_con_layer.shape

import scipy as sc


# resize image of ouput of last conv2d, using zoom
resize_image = sc.ndimage.zoom(output_con_layer, (int(224/output_con_layer.shape[0]),
                                                  int(224/output_con_layer.shape[1]), 1))

resize_image.shape

weights_for_predicted_class_for_this_image.shape

# array_shape(224*224, 0124) * array_shape(1024, 1) = array_shape(224*224, 1)
# final reshape (224*224, 1) to shape (224, 224, 1)
final_image = np.dot(
    resize_image.reshape(resize_image.shape[0] * resize_image.shape[1], resize_image.shape[2]),
    weights_for_predicted_class_for_this_image
).reshape(resize_image.shape[0], resize_image.shape[1])

final_image.shape

# heatmap image
plt.imshow(final_image, cmap='jet')
plt.figure(figsize=(3, 3))
plt.show()

img.shape

img_ = img.reshape(224, 224, 3)
img_.shape

from matplotlib.patches import Rectangle

np.unique(final_image)

final_image.shape

final_image = final_image/255

plt.figure(figsize = (4, 4))
plt.imshow(single_image)
plt.imshow(final_image, cmap='jet', alpha=0.3)

def getHeatMap(image):
  single_image = cv2.imread(image)
  single_image = cv2.resize(single_image, (224, 224))
  single_image = cv2.cvtColor(single_image, cv2.COLOR_BGR2RGB)
  img = single_image/255
  img = np.asarray(img)
  img = img.reshape(1, 224, 224, 3)
  y_pred = m.predict(img)[0]
  y_pred = np.argmax(y_pred)
  classes = list({'COVID': 0, 'Viral Pneumonia': 1})
  class_prediction = classes[y_pred]
  weights = m.layers[-1].get_weights()[0]
  weights = np.asarray(weights)
  weights = weights.reshape(weights.shape[1], weights.shape[0])
  weights_for_predicted_class_for_this_image = weights[y_pred]
  new_model = tf.keras.models.Model(
    m.input,
    m.get_layer('conv5_block16_concat').output
    )
  output_con_layer = new_model.predict(img)[0]
  resize_image = sc.ndimage.zoom(output_con_layer, (int(224/output_con_layer.shape[0]),
                                                  int(224/output_con_layer.shape[1]), 1))
  final_image = np.dot(
      resize_image.reshape(resize_image.shape[0]*resize_image.shape[1], resize_image.shape[2]),
      weights_for_predicted_class_for_this_image
  ).reshape(resize_image.shape[0], resize_image.shape[1])
  img_ = img.reshape(224, 224, 3)
  final_image = final_image/255
  return [img_, final_image, class_prediction]


datasetObject = pathlib.Path(os.path.join(dataset_folder))
images = list(datasetObject.glob("*/*/*.*"))
arr = []
for i in images[:8]:
    arr.append(getHeatMap(os.path.join(i)))

plt.figure(figsize = (20, 10))
for j, i in enumerate(arr[:8]):
  plt.subplot(2, 4, j + 1)
  plt.imshow(i[0])
  plt.title(i[2])
plt.show()

plt.figure(figsize=(20, 10))
for j, i in enumerate(arr[:8]):
    plt.subplot(2, 4, j + 1)
    plt.imshow(i[0])
    plt.imshow(i[1], cmap='jet', alpha=0.3)
    plt.title(i[2])
plt.show()

