import os, csv, shutil

path = '/Users/seungyoun/Desktop/ML/Dog_Bread_Identification'

f = open(path+'/labels.csv', 'r')
csvf = csv.reader(f)

labels = []
for i in csvf:
    labels.append(i)

def find_bread(name):
    for i in labels:
        if(i[0]==name):
            return i[1]

f.close()
images = os.listdir(path+'/train')

for img in images:
    bread = find_bread(img[:-4])
    if not os.path.exists(path+'/train/'+bread):
        os.makedirs(path+'/train/'+bread)
    src = path + '/train/' + img
    dest = path + '/train/' + bread +'/' + img
    shutil.move(src, dest)

print("DONE")
