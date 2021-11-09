import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image

model_json = {'VGG16': './model/VGG16model.json',
              'InceptionV3': './model/InceptionV3model.json',
              'ResNet101': './model/ResNet101model.json',
              'MobileNetV2': './model/MobileNetV2model.json',
              'EfficientNetB7 ': './model/EfficientNetB7model.json',
              }
model_h5 = {'VGG16': './model/VGG16model.h5',
            'InceptionV3': './model/InceptionV3model.h5',
            'ResNet101': './model/ResNet101model.h5',
            'MobileNetV2': './model/MobileNetV2model.h5',
            'EfficientNetB7 ': './model/EfficientNetB7model.h5',
            }
# 讀取模型json檔
json_file = open(model_json['MobileNetV2'], 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# 載入模型權重
model.load_weights(model_h5['MobileNetV2'])
# compile模型
base_learning_rate = 0.00001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

classes_name = ['dig', 'form', 'pouring', 'rebar']
path = './test'
allfilelist = os.listdir(path)
nf = open('predictions', 'a')
for file in allfilelist:
    img = Image.open('./test/' + str(file))
    img = img.resize((256, 256))
    # img = (np.expand_dims(img,0))
    # img = img.reshape(32,180,180,3)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    nf = open('MobileNetV2predictions', 'a')
    nf.write(
        "This " +
        str(file) + " most likely belongs to {} with a {:.2f} percent confidence.\n"
        .format(classes_name[np.argmax(score)], 100 * np.max(score))
    )
nf.close()
