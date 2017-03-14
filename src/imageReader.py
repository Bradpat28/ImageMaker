import glob
import os

#Returns arr of file names of given type in dir
def filesInDir(dir, type):
    os.chdir(dir)
    return glob.glob("*." + type)

arr = filesInDir("./images/", "jpg")

for entry in arr:
    print(entry)
