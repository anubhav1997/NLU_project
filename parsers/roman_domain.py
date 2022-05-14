import numpy as np
import pandas as pd

news_docs = ['Agenda-1-300', 'Agenda-b1', 'Agenda-b2', 'Agenda-b3']

med_docs = ['EMEA-b1', 'EMEA-b2', 'EMEA-b3', 'EMEA-noi-b1', 'Medical-1', 'Medical-2']

def parse_domain(in_file, docs):
    with open(in_file) as f:
        lines = f.readlines()
    new_lines = [line for line in lines if not 'sent_id' in line and not 'text =' in line]
    sentence = []
    tags = []
    sentences = []
    pos = []
    flag = False
    for line in new_lines:
        if 'newdoc' in line:
            if any(doc in line for doc in docs):
                flag = True
            else:
                flag = False
        else:
            if line == '\n':
                if len(sentence) > 0:
                    sentences.append(' '.join(sentence))
                    pos.append(' '.join(tags))
                    sentence = []
                    tags = []
            elif flag:
                tokens = line.split('\t')
                sentence.append(tokens[1])
                tags.append(tokens[3])

    df = pd.DataFrame()
    df['sentence'] = np.array(sentences).T
    df['tags'] = np.array(pos).T
    return df

df = parse_domain('ro_rrt-ud-train.conllu', news_docs)
df.to_csv('romanian_news.csv', index = False)