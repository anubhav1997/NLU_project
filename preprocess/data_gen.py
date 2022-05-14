from io import open
from conllu import parse_incr
import pandas as pd
import nltk
nltk.download('treebank')
nltk.download('brown')
nltk.download('conll2000')
nltk.download('universal_tagset')
nltk.download('indian')
nltk.download('sinica_treebank')
nltk.download('mac_morpho')
nltk.download('conll2002')
nltk.download('cess_cat')
nltk.download('universal_tagset')
nltk.download('cess_esp')
from nltk.corpus import brown, treebank, conll2000, indian, sinica_treebank, mac_morpho, conll2002, cess_cat,cess_esp
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader


data_file = open("/content/ud-treebanks-v2.9/UD_English-GUMReddit/en_gumreddit-ud-train.conllu", "r")

# for tokenlist in parse_incr(data_file):
#     print(tokenlist)

def read_conll(in_file, lowercase=False, max_example=None):
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:
                examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
                word, pos, head, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
    return examples

def get_lang_data():
    treebank_corpus = treebank.tagged_sents(tagset='universal')
    brown_corpus = brown.tagged_sents(tagset='universal', categories = 'news')
    conll_corpus = conll2000.tagged_sents(tagset='universal')
    indian_corpus = indian.tagged_sents()
    sinica_treebank_corpus = sinica_treebank.tagged_sents()
    conll_2002_corpus = conll2002.tagged_sents()
    mac_morpho_corpus = mac_morpho.tagged_sents()
    cess_cat_corpus = cess_cat.tagged_sents()
    tagged_sentences = treebank_corpus + brown_corpus + conll_corpus + indian_corpus + sinica_treebank_corpus + mac_morpho_corpus + cess_cat_corpus

    return tagged_sentences

cess_esp._tagset = "zh-sinica"

# nltk.tag.mapping._load_universal_map("zh-sinica")  # initialize; normally loaded on demand
mapdict = nltk.tag.mapping._MAPPINGS["zh-sinica"]["universal"] # shortcut to the map
print(mapdict)
# alltags = set(t for w, t in cess_esp.tagged_words())
# for tag in alltags:
#     if len(tag) <= 2:   # These are complete
#         continue
#     mapdict[tag] = mapdict[tag[:2]]

mapdict ={}
with open("sinica.txt") as f:
    lines  = f.readlines()
    for line in lines:
        line = line.strip()
        items = line.split("\t")
        # items = [x.split(" ") for x in line.split("\t")]
        # items = [item for sublist in items for item in sublist]
        mapdict[items[0].lower()] = items[1]
        
sinicab = sinica_treebank.tagged_sents()
new_sini = []
for i, row in enumerate(sinicab):
    new_word_list = []
    for j,word in enumerate(row):
        sinicab[i][j] = list(sinicab[i][j])
        #print(sinica_treebank_corpus[i][j])
        sinicab[i][j][1] = mapdict[word[1].lower()]
        new_word_list.append(sinicab[i][j])
        #print(new_word_list)
        new_sini.append(new_word_list)
        #print(new_sini)
        
dataset = get_lang_data()

cass_dict ={}
with open("spanish.txt") as f:
    lines  = f.readlines()
    for line in lines:
        line = line.strip()
        items = line.split("\t")
        # items = [x.split(" ") for x in line.split("\t")]
        # items = [item for sublist in items for item in sublist]
        cass_dict[items[0].lower()] = items[1]
        
cessa = cess_cat.tagged_sents()
new_sini_cass = []
count = 0
temp = []
anomaly = {}

for i,row in enumerate(cessa):
    new_word_list = []
    for j,word in enumerate(row):
        cessa[i][j] = list(cessa[i][j])
        #print(sinica_treebank_corpus[i][j])
        try:
            cessa[i][j][1] = cass_dict[word[1].lower()]
        except:
            cessa[i][j][1] = "UNK"
            # print(cessa[i][j][1])
            count += 1
            temp.append(cessa[i][j][1])

        new_word_list.append(cessa[i][j])
    #print(new_word_list)
    new_sini_cass.append(new_word_list)
    
dutch_dict ={}
with open("spanish.txt") as f:
    lines  = f.readlines()
    for line in lines:
        line = line.strip()
        items = line.split("\t")
        # items = [x.split(" ") for x in line.split("\t")]
        # items = [item for sublist in items for item in sublist]
        dutch_dict[items[0].lower()] = items[1]
        
dutch = conll2002.tagged_sents()
print(dutch[0])
new_sini_dutch = []
count = 0
temp = []
anomaly = {}
for i,row in enumerate(dutch):
    new_word_list = []
    for j,word in enumerate(row):
        dutch[i][j] = list(dutch[i][j])
        #print(sinica_treebank_corpus[i][j])
        try:
            dutch[i][j][1] = dutch_dict[word[1].lower()]
        except:
            dutch[i][j][1] = "UNK"
            # print(cessa[i][j][1])
            count += 1
            temp.append(dutch[i][j][1])

        new_word_list.append(dutch[i][j])
    # print(new_word_list)
    new_sini_dutch.append(new_word_list)
    
dutch = conll2002.tagged_sents()
print(dutch[40])

portu = mac_morpho.tagged_sents()
print(portu[0])

portu = mac_morpho.tagged_sents()
new_sini_portu = []
count = 0
temp = []
anomaly = {}

for i,row in enumerate(portu):
    new_word_list = []
    for j,word in enumerate(row):
        portu[i][j] = list(portu[i][j])
        #print(sinica_treebank_corpus[i][j])
        try:
            portu[i][j][1] = portu_dict[word[1].lower()]
        except:
            portu[i][j][1] = "UNK"
            # print(cessa[i][j][1])
            count += 1
            temp.append(portu[i][j][1])

        new_word_list.append(portu[i][j])
        #print(new_word_list)
    new_sini_portu.append(new_word_list)
    
ind_dict ={}
with open("indian.txt") as f:
    lines  = f.readlines()
    for line in lines:
        line = line.strip()
        items = line.split(" ")
        print(items)
        # items = [x.split(" ") for x in line.split("\t")]
        # items = [item for sublist in items for item in sublist]
        ind_dict[items[0].lower()] = items[-1]
        
ind = indian.tagged_sents()
new_sini_ind = []
count = 0
temp = []
anomaly = {}
for i,row in enumerate(ind):
    new_word_list = []
    for j,word in enumerate(row):
        ind[i][j] = list(ind[i][j])
        #print(sinica_treebank_corpus[i][j])
        try:
            ind[i][j][1] = ind_dict[word[1].lower()]
        except:
            ind[i][j][1] = "UNK"
            # print(cessa[i][j][1])
            count += 1
            temp.append(ind[i][j][1])
        new_word_list.append(ind[i][j])
        #print(new_word_list)
    new_sini_ind.append(new_word_list)
    
treebank_corpus = treebank.tagged_sents(tagset='universal')
brown_corpus = brown.tagged_sents(tagset='universal', categories = 'news')
conll_corpus = conll2000.tagged_sents(tagset='universal')

dataset = new_sini_portu + new_sini_cass + new_sini_dutch + new_sini_ind + new_sini + treebank_corpus + brown_corpus + conll_corpus

dataset_shuffled = shuffle(dataset)

total_data = len(dataset_shuffled)
train_data = int(0.75*total_data)
dev_test = train_data + int(0.10*total_data)

train_data_1 = dataset_shuffled[0:train_data]
dev_data = dataset_shuffled[train_data:dev_test]
test_data = dataset_shuffled[dev_test :]

def make_data(dataset):
    words = []
    tags_list = []
    for row in dataset:
        flag = 0
        count = 0
        for values in row:
            if(values[1] == 'UNK'):
                flag = 1
                count += 1
            if(count > 5):
                break
    if(count > 5):
        for values in row:
            words.append(values[0])
            tags_list.append(values[1])
        words.append('-DOCSTART-')
        tags_list.append('O')
    return words, tags_list


words_train, tag_train = make_data(train_data_1)
words_dev, tag_dev = make_data(dev_data)
words_test, tag_test = make_data(test_data)


df = pd.DataFrame(words_train, columns = ['word'])
df_2 = pd.DataFrame(tag_train, columns = ["tag"])
df_dev = pd.DataFrame(words_dev, columns = ['word'])
df_dev_tag= pd.DataFrame(tag_dev, columns = ["tag"])
df_test = pd.DataFrame(words_test, columns = ['word'])
df_test_tag= pd.DataFrame(tag_test, columns = ["tag"])


df.to_csv("train_x.csv",index = False)
df_2.to_csv("train_y.csv", index = False)
df_dev.to_csv("tdev_x.csv",index = False)
df_dev_tag.to_csv("dev_y.csv", index = False)
df_test.to_csv("test_x.csv",index = False)
df_test_tag.to_csv("test_y.csv", index = False)