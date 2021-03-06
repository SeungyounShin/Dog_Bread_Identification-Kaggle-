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
from tqdm import tqdm
import pandas as pd
import cv2

start = time.time()
path = '/Users/seungyoun/Desktop/ML/Dog_Bread_Identification'
test_ = pd.read_csv(path+'/kaggle_submission.csv')
train_ = pd.read_csv(path+'/labels.csv')

targets_series = pd.Series(train_['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)

#test data load
x_test = []

ff = os.listdir(path+'/test')
lst = []
for i in ff:
    lst.append(path+'/test/'+i)

i=0
for f in lst:
    img = load_img(f)
    x = img_to_array(img)  #  (3,img_w,img_h)
    tmp = scipy.misc.imresize(x,(224,224,3))
    tmp = tmp/255
    x_test.append(tmp)
    i +=1
    print(i,"images in list" )


print("[data preprocessing] it takes times\n ...")
x_test  = np.array(x_test, np.float32) / 255.

print(x_test.shape)

#model
json_file = open('resnet50.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(path+"/resnet50_DogBread_weight_3.h5")
print("Loaded model from disk\n"*3)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#make predictions
preds = model.predict(x_test, verbose=1)


#write submission file
print("Writing submission file")
sub = pd.DataFrame(preds)
# Set column names to those generated by the one-hot encoding earlier
col_names = one_hot.columns.values
sub.columns = col_names
# Insert the column id from the sample_submission at the start of the data frame
sub.insert(0, 'id', test_['id'])
sub.head(5)
sub.to_csv(path)

print("==FINISHED==")
print("It took ",time.time()-start,"seconds")
