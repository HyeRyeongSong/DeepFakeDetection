import os
import random
ri = random.randint

DIR = "."

Folder = [ foldername for foldername in os.listdir(DIR) if foldername.find('_') != -1 ]


for folder in Folder:
    mp4s = [ filename for filename in os.listdir("./" + folder) if filename[-4:] == '.mp4' ]
    Len = len(mp4s)
    li = getNumbers(10, Len)
    for mv in li:
        print(mp4s[mv])
        os.system(f'cp ./{folder}/{mp4s[mv]} ../mini_test')

def getNumbers(n, maxLen):
    li = []
    ran_num = random.randint(0,maxLen)

    for i in range(n):
        while ran_num in li:
            ran_num = random.randint(0,maxLen)
        li.append(ran_num)

    li.sort()
    return li