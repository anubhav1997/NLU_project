import os
import numpy as np
import pandas as pd

path = 'SPACCC_POS/corpus/tagged'

files = sorted([file for file in os.listdir(path) if file.endswith('.txt_tagged')])

dfs = []

for file in files:
    sentences = []
    pos = []
    lines = open(os.path.join(path, file)).readlines()
    sentence = []
    tags = []
    for line in lines:
        tokens = line.split(' ')
        if len(tokens) > 1:
            sentence.append(tokens[0])
            tags.append(tokens[2])
            if tokens[0] == '.':
                sentences.append(' '.join(sentence))
                pos.append(' '.join(tags))
                sentence = []
                tags = []
    
    df = pd.DataFrame()
    df['sentence'] = np.array(sentences).T
    df['tags'] = np.array(pos).T
    dfs.append(df)

final_df = pd.concat(dfs)

final_df.to_csv('spanish_clinical.csv', index = False)

