import re
import random
import emoji
from tqdm.contrib import tzip

train_label = []
train_text = []
post=[]
postBegin=False
nclass = {'negative': 0, 'neutral': 1, 'positive': 2}

for line in open('', 'r'):
    line = line.strip()
    c = re.split(r"\s+", line, maxsplit=2)
    if line[:4] == 'meta' and len(c) == 3:
        postBegin = True
        a = re.split(r"\s+", line, maxsplit=2)
        label = nclass[a[2]]
        post.append(label)
        continue

    if len(line)==0:
        postBegin=False
        LABEL =post[0]
        flag = False
        TEXT = ''
        #TEXT = ' '.join(i for i in post[1:])
        for i in post[1:]:
            if '@' in i:
                flag=True
                continue
            if flag==True:
                flag=False
                continue
            if '…' in i:
                i ='.'
            if '_' in i:
                continue
            if 'http' in i:
                break
            TEXT = TEXT+' '+i
        TEXT = emoji.demojize(TEXT)
        train_label.append(LABEL)
        train_text.append(TEXT)
        post=[]
        continue
    if postBegin:
        b = re.split(r"\s+", line, maxsplit=1)
        text = b[0]
        post.append(text)


F1 = open('', 'w')
traintext = ''
vocab = []
for text, label in tzip(train_text, train_label):
        x1 = list(text)
        y = label
        traintext = ''.join(str(i) for i in x1)
        traintext = '    '+ traintext
        traintext = str(y) + traintext+'\n'
        F1.write(traintext)
F1.close()

