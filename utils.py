import re
import torch
import collections
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')  #

# label_vocab = collections.defaultdict(lambda: len(label_vocab))
# label_vocab['<pad>'] = 0


class PosTaggingDataset(Dataset):
    def __init__(self, sentences, taggings):
        assert len(sentences) == len(taggings)
        self.sentences = sentences
        self.taggings = taggings

    def __getitem__(self, i):
        return self.sentences[i], self.taggings[i]

    def __len__(self):
        return len(self.sentences)


def collate_fn(items):
    max_len = max(len(item[0]) for item in items)

    sentences = torch.zeros((len(items), max_len), device=items[0][0].device).long().to(device)
    taggings = torch.zeros((len(items), max_len)).long().to(device)

    for i, (sentence, tagging) in enumerate(items):
        sentences[i][0:len(sentence)] = sentence
        taggings[i][0:len(tagging)] = tagging

    return sentences, taggings


def get_dataloader(sentences, tags, model_name, tokenizer, label_vocab, test_size=0.25, batch_size=64):
    train_sentences, valid_sentences, train_taggings, valid_taggings = train_test_split(sentences, tags, test_size=0.25,
                                                                                        random_state=42)

    train_bert_tokenized_sentences, train_aligned_taggings = align_tokenizations(train_sentences, train_taggings,
                                                                                 model_name, tokenizer)
    valid_bert_tokenized_sentences, valid_aligned_taggings = align_tokenizations(valid_sentences, valid_taggings,
                                                                                 model_name, tokenizer)
    test_bert_tokenized_sentences, test_aligned_taggings = align_tokenizations(valid_sentences, valid_taggings,
                                                                               model_name, tokenizer)

    # print(train_sentences[30], tags[30])
    # print(train_bert_tokenized_sentences[30], train_aligned_taggings[30])

    # print("1: ", len(label_vocab))
    train_sentences_ids, train_taggings_ids, label_vocab = convert_to_ids(train_bert_tokenized_sentences, train_aligned_taggings,
                                                             label_vocab, tokenizer)
    # print("2: ", len(label_vocab))
    valid_sentences_ids, valid_taggings_ids, _ = convert_to_ids(valid_bert_tokenized_sentences, valid_aligned_taggings,
                                                             label_vocab, tokenizer)
    # print("3: ", len(label_vocab))
    test_sentences_ids, test_taggings_ids, _ = convert_to_ids(test_bert_tokenized_sentences, test_aligned_taggings,
                                                           label_vocab, tokenizer)
    # print("4: ", len(label_vocab))


    # print(np.array(train_taggings_ids).shape)
    # print(np.array(train_sentences_ids).shape)

    train_loader = DataLoader(PosTaggingDataset(train_sentences_ids, train_taggings_ids), batch_size=batch_size,
                              collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(PosTaggingDataset(valid_sentences_ids, valid_taggings_ids), batch_size=batch_size,
                              collate_fn=collate_fn)
    test_loader = DataLoader(PosTaggingDataset(test_sentences_ids, test_taggings_ids), batch_size=batch_size,
                             collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader, label_vocab


def convert_to_ids(sentences, taggings, label_vocab, tokenizer):
    sentences_ids = []
    taggings_ids = []
    for sentence, tagging in zip(sentences, taggings):
        sentence_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(['[CLS]'] + sentence + ['SEP'])).long()
        tagging_tensor = torch.tensor([0] + [label_vocab[tag] for tag in tagging] + [0]).long()

        sentences_ids.append(sentence_tensor.to(device))
        taggings_ids.append(tagging_tensor.to(device))
    return sentences_ids, taggings_ids, label_vocab


def align_tokenizations(sentences, taggings, model_name, tokenizer):
    bert_tokenized_sentences = []
    aligned_taggings = []
    count = 0
    for sentence, tagging in zip(sentences, taggings):
        # first generate BERT-tokenization
        #     try:
        #     print("HEREEEE")
        bert_tokenized_sentence = tokenizer.tokenize(' '.join(sentence))
        #     except:
        #       print("Error",sentence)
        #     print(bert_tokenized_sentence, sentence)
        aligned_tagging = []
        current_word = ''

        index = 0  # index of current word in sentence and tagging
        for token in bert_tokenized_sentence:
            if model_name == 'bert' or model_name == 'bert_adapter':

                current_word += re.sub(r'^##', '', token)  # recompose word with subtoken
            #       current_word += re.sub(r'▁', '', token) # recompose word with subtoken
            elif model_name == 'xlm' or model_name == 'bert_adapter':

                current_word += re.sub(r'</w>', '', token)  # recompose word with subtoken

            # print(index, token, len(sentence)) #, sentence)
            try:
                sentence[index] = sentence[index].replace('\xad', '')  # fix bug in data
            except:
                print(sentence)
                break
                # print(bert_tokenized_sentence)

            # note that some word factors correspond to unknown words in BERT
            # print(token, sentence[index].startswith(current_word), sentence[index], current_word)
            # assert token == '[UNK]' or sentence[index].startswith(current_word)

            if token == '[UNK]' or sentence[index] == current_word:  # if we completed a word
                current_word = ''
                aligned_tagging.append(tagging[index])
                index += 1
            else:  # otherwise insert padding
                aligned_tagging.append('<pad>')

        # assert len(bert_tokenized_sentence) == len(aligned_tagging)

        if (len(aligned_tagging) == len(bert_tokenized_sentence)):

            bert_tokenized_sentences.append(bert_tokenized_sentence)
            aligned_taggings.append(aligned_tagging)
        else:
            print(len(aligned_tagging), len(bert_tokenized_sentence))
            count += 1

    return bert_tokenized_sentences, aligned_taggings


def get_tag_count(train_taggings):
    # use a defaultdict to count the number of occurrences of each tag
    import collections
    tagset = collections.defaultdict(int)

    for tagging in train_taggings:
        for tag in tagging:
            tagset[tag] += 1

    print('number of different tags:', len(tagset))

    # print count and tag sorted by decreasing count
    for tag, count in sorted(tagset.items(), reverse=True, key=lambda x: x[1]):
        print(count, tag)
