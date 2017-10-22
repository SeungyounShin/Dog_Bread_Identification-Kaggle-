from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import model_from_json
import keras
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = './train'
nb_train_samples = 10222
epochs = 50
batch_size = 19

#image shape
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#model
resnet50 = keras.applications.resnet50.ResNet50(weights = None)
classes = 120
resnet50.layers.pop()
for layer in resnet50.layers:
    layer.trainable=False
last = resnet50.layers[-1].output
x = Dense(120, activation="softmax")(last)
model = Model(resnet50.input, x)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
print("Model loaded")

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs)


model_json = model.to_json()
with open("resnet50.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("resnet50_DogBread_weight.h5")
print("Saved model to disk")
