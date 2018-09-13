import os
from random import shuffle

path ="/media/tanmoy/3A763A067639C383/Users/Tanmoy/Desktop/tanmoydataset/Breast_Histo_partial"

dirList=os.listdir(path)
ara=[]
with open("/mnt/64220DD6220DADDC/Thesis/ThesisWork/Code/BreastData/image_name.txt", "w") as f:
    for path, subdirs, files in os.walk(path):
        for filename in files:
            temp=path + "/" +filename
            ara.append(temp)

    shuffle(ara)
    for name in ara:
        f.write(name + "\n")
