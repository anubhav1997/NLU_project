import sys
import numpy as np
import pandas as pd

sentences = []
pos = []

args = sys.argv

with open(args[1]) as f:
    lines = f.readlines()
    for line in lines:
        tokens = line.split(' ')
        sentence = []
        tags = []
        for token in tokens:
            if token and token != ' ' and token != '\n' and token != '\t':
                split = token.split('/')
                if len(split) > 1:
                    sentence.append(split[0])
                    tags.append(split[1])
        sentences.append(' '.join(sentence))
        pos.append(' '.join(tags))

df = pd.DataFrame()

df['sentence'] = np.array(sentences).T
df['tags'] = np.array(pos).T

df.to_csv('genia_english_medical.csv', index = False)
# df = pd.DataFrame()