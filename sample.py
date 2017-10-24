import os
from shutil import copyfile

project_root = '/Users/seungyoun/Desktop/ML/Dog_Bread_Identification'
path = '/Users/seungyoun/Desktop/ML/Dog_Bread_Identification/train'
dst_path = project_root+'/repre'

tmp = os.listdir(path)

tmp = tmp[1:]

for i in tmp:
    select = os.listdir(path+'/'+i)[8]
    copyfile(path+'/'+i+'/'+select, dst_path+'/'+i+'.jpg')
