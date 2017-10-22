from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.models import Sequential
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing import image
import numpy as np
import os,time,ssl,scipy.misc
import matplotlib.image as img
ssl._create_default_https_context = ssl._create_unverified_context

#basic setting
img_width, img_height = 224, 224

path = '/Users/seungyoun/Desktop/ML/Dog_Bread_Identification'
breeds = os.listdir(path+'/train')[1:]
nb_train_samples = 10222

#model
# load json and create model
json_file = open('resnet50.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(path+"/resnet50_DogBread_weight_3.h5")
print("Loaded model from disk")
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

def predict(path):
    img = load_img(path)
    x = img_to_array(img)  #  (3,img_w,img_h)
    tmp = scipy.misc.imresize(x,(224,224,3))
    tmp = tmp/255
    z = np.reshape(tmp,(1,224,224,3))

    # evaluate loaded model on test data
    pred = model.predict(z)

    return pred

pred = np.argmax(predict(path+'/me3.jpg'))
print(breeds[pred])
