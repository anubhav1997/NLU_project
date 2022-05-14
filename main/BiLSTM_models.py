

import torch.nn as nn
import torch
import torch.nn.functional as F


from torchtext import data

import numpy as np


def get_one_batch(tokens, vocab, alphabet, tags, all_tags):  # ):
    def token_to_char_tensor(tok):
        tensor = torch.LongTensor([alphabet.stoi[c] for c in tok])
        return tensor

    tokens_tensor = torch.LongTensor([vocab.stoi[tok] for tok in tokens]).to(device)
    # Perform padding for tokens
    max_tok_len = max([len(tok) for tok in tokens])
    # print(max_tok_len)
    tokens_as_char = [token_to_char_tensor(tok) for tok in tokens]
    for idx, tok in enumerate(tokens_as_char):
        if len(tok) < max_tok_len:
            paddings = torch.from_numpy(np.full(max_tok_len - len(tok), alphabet.stoi['<pad>']))
            tok = torch.cat((tok, paddings))
        tokens_as_char[idx] = tok
    char_tensor = torch.stack([c_tnsr for c_tnsr in tokens_as_char], dim=1).to(device)

    tags_tensor = []
    for tag in tags:
        try:
            tags_tensor.append(all_tags.index(tag))
        except:
            tags_tensor.append(0)

    #     tags_tensor = torch.LongTensor([all_tags.index(tag) for tag in tags]).to(device)
    tags_tensor = torch.LongTensor(tags_tensor).to(device)
    return tokens_tensor, char_tensor, tags_tensor




def build_vocab_from_sentences_tokens(sentences_tokens):
    """
    use torch text to build vocab object from a list of sentences that is already tokenized in to tokens
    :param sentences_tokens: list of list of tokens
    :return: torchtext.vocab object
    """
    token_field = data.Field(tokenize=list, init_token='<root>')
    fields = [('tokens', token_field)]
    examples = [data.Example.fromlist([t], fields) for t in sentences_tokens]
    torch_dataset = data.Dataset(examples, fields)
    token_field.build_vocab(torch_dataset)
    return token_field.vocab

# alphabet = build_alphabet_from_sentence_tokens(train_sentences)
# vocab = build_vocab_from_sentences_tokens(train_sentences)
# tokens_tensor, char_tensor, tags_tensor = get_one_batch(train_sentences[1], vocab, alphabet, train_taggings[1],
#                                                         list(tagset))



def build_alphabet_from_sentence_tokens(sentences_tokens):
    """
    Build alphabet from tokens by converting tokens to character
    :param sentences_tokens:
    :return:
    """
    def to_char(tokens):
      # try:
      return [c for tok in tokens for c in list(tok)]
      # except:
      #   print("Error")
    sentences_char = [to_char(sent) for sent in sentences_tokens]
    # print(sentences_char[0])

    char_field = data.Field(tokenize=list, init_token='<root>')
    fields = [('char', char_field)]
    examples = [data.Example.fromlist([t], fields) for t in sentences_char]
    # print(examples[0])
    torch_dataset = data.Dataset(examples, fields)
    char_field.build_vocab(torch_dataset)
    return char_field.vocab



class CustomedBiLstm(nn.Module):
    def __init__(self,
                 alphabet_size,
                 vocab_size,
                 word_embed_dim,
                 char_embed_dim,
                 char_hidden_dim,
                 word_hidden_dim,
                 n_tags,
                 use_gpu):
        super(CustomedBiLstm, self).__init__()
        self.alphabet_size = alphabet_size
        self.vocab_size = vocab_size
        self.word_embed_dim = word_embed_dim
        self.char_embed_dim = char_embed_dim
        self.char_hidden_dim = char_hidden_dim
        self.word_hidden_dim = word_hidden_dim
        self.n_tags = n_tags

        self.char_embedding_layer = nn.Embedding(self.alphabet_size, self.char_embed_dim)
        self.lower_LSTM = nn.LSTM(input_size=self.char_embed_dim,
                                  hidden_size=self.char_hidden_dim,
                                  batch_first=True)

        self.word_embedding_layer = nn.Embedding(self.vocab_size, self.word_embed_dim)
        self.upper_LSTM = nn.LSTM(input_size=self.char_hidden_dim + self.word_embed_dim,
                                  hidden_size=self.word_hidden_dim,
                                  bidirectional=True)
        self.hidden_to_tag = nn.Linear(self.word_hidden_dim * 2, self.n_tags)

    def forward(self, tokens_tensor, char_tensor):
        char_embeds = self.char_embedding_layer(char_tensor)
        lower_lstm_out, hidden = self.lower_LSTM(char_embeds)
        last_state_lower_lstm = lower_lstm_out[-1]
        tokens_embeds = self.word_embedding_layer(tokens_tensor)

        final_embeds = torch.cat((last_state_lower_lstm, tokens_embeds), 1).view(tokens_tensor.shape[0], 1, -1)
        upper_lstm_out, hidden = self.upper_LSTM(final_embeds)
        out = self.hidden_to_tag(upper_lstm_out.view(tokens_tensor.shape[0], -1))
        out = F.log_softmax(out, dim=1)
        return out

