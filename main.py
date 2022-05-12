import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import collections
import pickle
import argparse
from transformers import AutoConfig
from transformers import AutoAdapterModel
from sklearn.model_selection import train_test_split
from trainer import fit, fit_intermixing, perf, fit_interleaving
from transformer_models import get_tokenizer, LinearProbeBert_adapter, LinearProbeXLM_adapter, LinearProbeXLM, LinearProbeBERT
from utils import get_dataloader

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', default='bert', type=str)
parser.add_argument('--test_dataset', default='', type=str)
parser.add_argument('--full_test_model_path', default='', type=str)
parser.add_argument('--mode', default='standard', type=str)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--data_path', default="/home/aj3281/NLU_project/", type=str)


# steps = 2000
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
config = AutoConfig.from_pretrained(
    "bert-base-multilingual-cased",
)

with open(args.data_path + 'train_words.pkl', 'rb') as f:
    sentences = pickle.load(f)
with open(args.data_path + 'train_tags.pkl', 'rb') as f:
    tags = pickle.load(f)


with open(args.data_path + 'german_train_words.pkl', 'rb') as f:
    sentences_german = pickle.load(f)
with open(args.data_path + 'german_train_tags.pkl', 'rb') as f:
    tags_german = pickle.load(f)


with open(args.data_path + 'medical_english_train_words.pkl', 'rb') as f:
    sentences_medical = pickle.load(f)
with open(args.data_path + 'medical_english_train_tags.pkl', 'rb') as f:
    tags_medical = pickle.load(f)


with open(args.data_path + 'romanian_news_words.pkl', 'rb') as f:
    sentences_rom = pickle.load(f)
with open(args.data_path + 'romanian_news_tags.pkl', 'rb') as f:
    tags_rom = pickle.load(f)


label_vocab = collections.defaultdict(lambda: len(label_vocab))
label_vocab['<pad>'] = 0


if args.mode != 'intermixing':
    BATCH_SIZE = 64
else:
    BATCH_SIZE = 32





tokenizer = get_tokenizer(args.model_name)


# print()


model_name = args.model_name

if args.mode != 'shuffling':
    train_loader, valid_loader, test_loader, label_vocab = get_dataloader(sentences+sentences_german+sentences_rom, tags+tags_german+tags_rom, model_name, tokenizer, label_vocab, test_size=0.25, batch_size=BATCH_SIZE)
    # print("here 1", len(label_vocab))
    train_loader_medical, valid_loader_medical, test_loader_medical, label_vocab = get_dataloader(sentences_medical, tags_medical, model_name, tokenizer, label_vocab, test_size=0.1, batch_size=BATCH_SIZE)
    # print("here 2", len(label_vocab))
else:
    train_loader, valid_loader, test_loader, label_vocab = get_dataloader(sentences + sentences_german + sentences_rom + sentences_medical,
                                                             tags + tags_german + tags_rom + tags_medical, model_name, tokenizer, label_vocab, test_size=0.25,
                                                             batch_size=BATCH_SIZE)

if args.model_name == 'bert':
    model = LinearProbeBERT(len(label_vocab))
elif args.model_name == 'bert_adapter':
    model = LinearProbeBert_adapter(len(label_vocab))
elif args.model_name == 'xlm':
    model = LinearProbeXLM(len(label_vocab))
elif args.model_name == 'xlm_adapter':
    model = LinearProbeXLM_adapter(len(label_vocab))



# print(label_vocab)

if args.mode == 'standard':
    fit(model, args.epoch, train_loader, valid_loader, label_vocab, lr=args.lr)
elif args.mode == 'finetune':
    fit(model, args.epoch, train_loader, valid_loader, label_vocab, lr=args.lr)
    fit(model, args.epoch, train_loader_medical, valid_loader, label_vocab, lr=args.lr)
elif args.mode == 'shuffling':
    fit(model, args.epoch, train_loader, valid_loader, label_vocab, lr=args.lr)
elif args.mode == 'intermixing':
    fit_intermixing(model, args.epoch, train_loader, train_loader_medical, valid_loader, label_vocab, lr=args.lr)
elif args.mode == 'interleaving':
    fit_interleaving(model, args.epoch, train_loader, train_loader_medical, valid_loader, label_vocab, lr=args.lr)



torch.save(model, args.model_name + "_" + args.mode + ".pth")

with open(args.data_path + 'clinical_spanish_words.pkl', 'rb') as f:
    sentences_spanish_med = pickle.load(f)
with open(args.data_path + 'clinical_spanish_tags.pkl', 'rb') as f:
    tags_spanish_med = pickle.load(f)

train_loader_spanish_medical, valid_loader_spanish_medical, test_loader_spanish_medical, label_vocab = get_dataloader(sentences_spanish_med, tags_spanish_med, model_name, tokenizer, label_vocab, test_size=1.0, batch_size=BATCH_SIZE)
print('SPANISH MEDICAL', *perf(model, test_loader_spanish_medical, label_vocab))


with open(args.data_path + 'social_german_words.pkl', 'rb') as f:
    sentences_german_social = pickle.load(f)
with open(args.data_path + 'social_german_tags.pkl', 'rb') as f:
    tags_german_social = pickle.load(f)

train_loader_german_social, valid_loader_german_social, test_loader_german_social, label_vocab = get_dataloader(sentences_german_social, tags_german_social, model_name, tokenizer, label_vocab, test_size=1.0, batch_size=BATCH_SIZE)
print('GERMAN SOCIAL', *perf(model, test_loader_german_social, label_vocab))


with open(args.data_path + 'romanian_medical_words.pkl', 'rb') as f:
    sentences_romanian_medical = pickle.load(f)
with open(args.data_path + 'romanian_medical_tags.pkl', 'rb') as f:
    tags_romanian_medical = pickle.load(f)

train_loader_romanian_medical, valid_loader_romanian_medical, test_loader_romanian_medical, label_vocab = get_dataloader(sentences_romanian_medical, tags_romanian_medical, model_name, tokenizer, label_vocab, test_size=1.0, batch_size=BATCH_SIZE)

print('ROMANIAN MEDICAL', *perf(model, test_loader_romanian_medical, label_vocab))


print('ALL DATA', *perf(model, test_loader, label_vocab))

print('ENGLISH MEDICAL', *perf(model, test_loader_medical, label_vocab))



