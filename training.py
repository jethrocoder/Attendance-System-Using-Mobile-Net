import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, Sequential , load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout, Embedding, LSTM , Flatten , GlobalAveragePooling2D
from keras.layers.merge import add
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import os
import pickle

def train():
    model = Xception(weights="imagenet",input_shape=(100,100,3),include_top=False)
    layer = GlobalAveragePooling2D()(model.layers[-1].output)
    model = Model(inputs = [model.input],outputs=[layer])
    for i in range(len(model.layers)):
        model.layers[i].trainable = False

    def create_model(model,output_neurons):
        output_layer = Dense(units=256,activation='relu')(model.layers[-1].output)
        output_layer = Dropout(0.3)(output_layer)
        output_layer = Dense(units=output_neurons,activation='softmax')(output_layer)
        return Model(inputs = [model.input],outputs=[output_layer])

    dataset_path = './train/'
    output_neurons_cnt = len(os.listdir(dataset_path))
    model = create_model(model,output_neurons_cnt)
    adam = Adam()
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['acc'])
    model.summary()

    for i in range(len(model.layers)):
        print(model.layers[i].trainable,end= " ")

    with open('./saved/id_to_roll_f.pkl','rb') as f:
        try:
            id_to_roll = pickle.load(f)
        except EOFError:
            id_to_roll = {}
    with open('./saved/roll_to_id_f.pkl','rb') as f:
        try:
            roll_to_id = pickle.load(f)
        except EOFError:
            roll_to_id = {}


    print(len(id_to_roll))
    print(len(roll_to_id))

    X = []
    Y = []
    for folder in os.listdir('./train'):
        folder_path = './train/' + folder
        for file in os.listdir(folder_path):
            roll = folder
            file_path = folder_path + '/' + file
            img = image.load_img(file_path,target_size=(100,100))
            img = image.img_to_array(img)
            # img /= 256.0
            X.append(img)
            Y.append(roll_to_id[roll])

    Y = np_utils.to_categorical(Y)
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y.shape)

    checkpoint = ModelCheckpoint(
        './saved/model/face_recogizer_model.hd5',
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
    )

    model.fit(X,Y,shuffle=True,validation_split=0.2,epochs=100,callbacks= [checkpoint])
        
