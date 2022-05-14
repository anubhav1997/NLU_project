import pickle, json
import pandas as pd

train_x = pd.read_csv("../Downloads/train_x-3.csv")
train_y = pd.read_csv("../Downloads/train_y-3.csv")

list_train_x = []
for i in train_x["word"]:
    list_train_x.append(i)
train_x_new = set(list_train_x)

train_x_new = list(train_x_new)

list_train_y = []
for i in train_y["tag"]:
    list_train_y.append(i)
train_y_new = set(list_train_y)

train_y_new = list(train_y_new)

train_x_new[1:20]

with open('vocab.json', 'w', encoding='utf8') as json_file:
    json.dump(train_x_new, json_file, ensure_ascii=False)
    
with open('tags.json', 'w', encoding='utf8') as json_file:
    json.dump(train_y_new, json_file, ensure_ascii=False)
    
final_list_words = []
sentence_list = []
final_list_tags = []
tags_list = []

for i in range(len(train_x["word"])):
    if(train_x["word"][i] == "-DOCSTART-"):
        final_list_words.append(sentence_list)
        final_list_tags.append(tags_list)
        sentence_list = []
        tags_list = []
    elif (isinstance(train_x["word"][i], (int,float))):
          continue
    else:
        sentence_list.append(train_x["word"][i])
        tags_list.append(train_y["tag"][i])
        
final_list_tags = []
tags_list = []
for i in train_y["tag"]:
    if(i == "O"):
        final_list_tags.append(tags_list)
        tags_list = []
    else:
        tags_list.append(i)
        
with open('train_words.pkl', 'wb') as f:
    pickle.dump(final_list_words, f)
    
with open('train_tags.pkl', 'wb') as f:
    pickle.dump(final_list_tags, f)