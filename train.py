# Importing all the required libraries
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, LeakyReLU, GlobalAveragePooling2D    
from tensorflow.keras import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json, clone_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from classification_models.tfkeras import Classifiers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", help="Train folder path where all the classes folders are present", type=str)
parser.add_argument("-v", help="Validation folder path where all the classes folders are present", type=str)
parser.add_argument("-c", nargs="+", default=["luad", "lusc"]) # Classes
parser.add_argument("-ps", help="Path where the model files hsould be saved", type=str)
parser.add_argument("-s", help="Name of the json/weight file which should be used while saving", type=str)
parser.add_argument("-n", help="Number of epochs", type=int, default=10)

args = parser.parse_args()

t, v, c, ps, s, n = args.t, args.v, args.c, args.ps, args.s, args.n

if not t:
    raise("Train path not specified")

if not v:
    raise("Validatino path not specified")

if not ps:
    raise("Model saving path not specified")

if not s:
    raise("Model saving name not specified")

# Shortcut without convolutional layer
def iden(X, filters):

    f1, f2, f3 = filters

    X_short = X

    X = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding="valid", kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), padding="valid", kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3)(X)

    X = Add()([X_short, X])
    X = Activation("relu")(X)

    return X

# Shortcut with convolutional layer
def conv(X, filters):

    f1, f2, f3 = filters

    X_short = X

    X = Conv2D(f1, kernel_size=(1, 1), strides=(2, 2), padding="valid", kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), padding="valid", kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X_short = Conv2D(f3, kernel_size=(1, 1), strides=(2, 2), padding="valid", kernel_initializer=glorot_uniform())(X_short)
    X_short = BatchNormalization(axis=3)(X_short)

    X = Add()([X, X_short])
    X = Activation("relu")(X)

    return X

def myMod(input_size=(224, 224, 3), classes=["luad", "lusc"]):

    X_input = Input(input_size)

    X = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), kernel_initializer=glorot_uniform())(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2), (2, 2))(X)
    
    drop = 0
    
    X = conv(X, [64, 64, 256])
    X = Dropout(drop)(X)
    #X = iden(X, [64, 64, 256])
    #X = Dropout(drop)(X)
    X = conv(X, [64, 64, 256])
    X = Dropout(drop)(X)
    #X = iden(X, [64, 64, 256])
    #X = Dropout(drop)(X)

    X = conv(X, [128, 128, 512])
    X = Dropout(drop)(X)
    #X = iden(X, [128, 128, 512])
    #X = Dropout(drop)(X)
    X = conv(X, [128, 128, 512])
    X = Dropout(drop)(X)
    #X = iden(X, [128, 128, 512])
    #X = Dropout(drop)(X)

    X = conv(X, [256, 256, 1024])
    X = Dropout(drop)(X)
    #X = iden(X, [256, 256, 1024])
    #X = Dropout(drop)(X)
    X = conv(X, [256, 256, 1024])
    X = Dropout(drop)(X)
    #X = iden(X, [256, 256, 1024])
    #X = Dropout(drop)(X)
    
    X = Flatten()(X)
#     X = Dense(2048, activation="relu", kernel_initializer=glorot_uniform())(X)
#     X = Dropout(0.4)(X)
#     X = Dense(2048, activation="relu", kernel_initializer=glorot_uniform())(X)
#     X = Dropout(0.4)(X)
    X = Dense(2048, kernel_initializer=glorot_uniform())(X)
    X = LeakyReLU()(X)
    X = Dropout(0.6)(X)
    X = Dense(2048, kernel_initializer=glorot_uniform())(X)
    X = LeakyReLU()(X)
    X = Dropout(0.6)(X)
    X = Dense(len(classes), activation="softmax", kernel_initializer=glorot_uniform())(X)

    model = Model(X_input, X)

    model.summary()

    return model

# Loading the model
model = myMod(classes=c)

# SeResNet18, preprocess_input = Classifiers.get('seresnet18')
# model.add(SeResNet18((224, 224, 3), weights='imagenet', include_top=False))
# model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.6))
# model.add(Dense(1000, activation='relu', kernel_initializer = glorot_uniform()))
# model.add(Dropout(0.6))
# model.add(Dense(len(c), activation='softmax'))
# model.summary()

batch_size = 32

datagen1 = ImageDataGenerator(rescale=1/255)
datagen2 = ImageDataGenerator(rescale=1/255)

# Generating the train and validation generators
train = datagen1.flow_from_directory(t, target_size=(224, 224), batch_size=batch_size, classes=c, class_mode="categorical")
validation= datagen2.flow_from_directory(v, target_size=(224, 224), batch_size=1, classes=c, class_mode="categorical", shuffle=False)

# Specifying the learning rate and optimizer
init = 1e-3
lrate = ExponentialDecay(init, decay_steps=train.n//batch_size*100, decay_rate=0.99, staircase=False)
opt = Adam(learning_rate=lrate)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Setting the callback functino to save the model after each epoch
ps = ps if ps[-1]=="/" else ps+"/"

model_json = model.to_json()
with open(ps+s+".json", "w") as json_file:
    json_file.write(model_json)

cp_callback = ModelCheckpoint(filepath=ps+s+"{epoch}.h5", save_weights_only=True,
                                                 #save_freq=train.n//(10*batch_size),
                                                 verbose=1)

# Training the model
print(model.fit(train, steps_per_epoch=train.n//batch_size, validation_data=validation, validation_steps=validation.n, epochs=n, max_queue_size=200, workers=200, callbacks=[cp_callback]))