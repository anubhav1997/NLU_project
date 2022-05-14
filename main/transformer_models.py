
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel,  AutoAdapterModel
from transformers import AutoConfig

# from transformers import AutoModelForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
config = AutoConfig.from_pretrained(
    "bert-base-multilingual-cased",
)


def get_tokenizer(model_name):
    if model_name=='xlm' or model_name == 'xlm_adapter':
        # load tokenizer for a specific bert model (bert-base-cased)
        tokenizer = AutoTokenizer.from_pretrained('xlm-mlm-100-1280')  # ('bert-base-multilingual-cased')
    elif model_name == 'bert' or model_name == 'bert_adapter':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    return tokenizer



class LinearProbeXLM(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained('xlm-mlm-100-1280')  # ('bert-base-multilingual-cased')
        self.probe = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.to(device)

    def parameters(self):
        return self.probe.parameters()

    def forward(self, sentences):
        outputs = self.bert(sentences)
        #     with torch.no_grad(): # no training of BERT parameters
        #         word_rep, sentence_rep = self.bert(sentences, return_dict=False)
        return self.probe(outputs.last_hidden_state)


class LinearProbeBERT(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-multilingual-cased')
        self.probe = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.to(device)

    def parameters(self):
        return self.probe.parameters()

    def forward(self, sentences):
        outputs = self.bert(sentences)
        #     with torch.no_grad(): # no training of BERT parameters
        #         word_rep, sentence_rep = self.bert(sentences, return_dict=False)
        return self.probe(outputs.last_hidden_state)



class LinearProbeBert_adapter(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        #     self.bert = AutoModel.from_pretrained('bert-base-multilingual-cased')
        self.bert = AutoAdapterModel.from_pretrained(
            "bert-base-multilingual-cased",
            config=config,
        )
        self.probe = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.to(device)

    def parameters(self):
        return self.probe.parameters()

    def forward(self, sentences):
        with torch.no_grad():  # no training of BERT parameters
            word_rep, sentence_rep = self.bert(sentences, return_dict=False)
        return self.probe(word_rep)



class LinearProbeXLM_adapter(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        #     self.bert = AutoModel.from_pretrained('bert-base-multilingual-cased')
        self.bert = AutoAdapterModel.from_pretrained(
            "xlm-mlm-100-1280",
            config=config,
        )
        self.probe = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.to(device)

    def parameters(self):
        return self.probe.parameters()

    def forward(self, sentences):
        with torch.no_grad():  # no training of BERT parameters
            word_rep, sentence_rep = self.bert(sentences, return_dict=False)
        return self.probe(word_rep)


