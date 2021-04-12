import os
import random

filenames = os.listdir("data/SegmentationClass")
filenames = [f.replace('.png', '\n') for f in filenames ]

random.shuffle(filenames)

l = len(filenames)

train=sorted(filenames[:8*int(l/10)])
val=sorted(filenames[8*int(l/10):9*int(l/10)])
test=sorted(filenames[9*int(l/10):])

with open('data/train.txt', 'w') as f:
    f.writelines(train)
with open('data/val.txt', 'w') as f:
    f.writelines(val)
with open('data/test.txt', 'w') as f:
    f.writelines(test)