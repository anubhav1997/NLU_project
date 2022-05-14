
import torch.optim as optim
from itertools import cycle
from transformers import AutoTokenizer, AutoModel

import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


def fit(model, epochs, train_loader, valid_loader, label_vocab, lr=1e-2, alphabet=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = num = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()  # start accumulating gradients
            if x.shape[1] >= 512:
                continue
            else:
                if (alphabet):
                    y_scores = model(x, alphabet[i])
                else:
                    y_scores = model(x)

                # print(y_scores.shape, y.shape)
                loss = criterion(y_scores.view(-1, len(label_vocab)), y.view(-1))
                if not torch.isnan(loss):
                    loss.backward()  # compute gradients though computation graph
                    optimizer.step()  # modify model parameters
                    total_loss += loss.item()
                num += 1
        print(1 + epoch, total_loss / num, *perf(model, valid_loader, label_vocab))


def perf(model, loader, label_vocab, alphabet=None):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # do not apply training-specific steps such as dropout
    total_loss = correct = num_loss = num_perf = 0
    num_loss = 0
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():  # no need to store computation graph for gradients
            # perform inference and compute loss
            if x.shape[1] >= 512:
                continue
            if alphabet:
                y_scores = model(x, alphabet[i])
            else:
                y_scores = model(x)
            loss = criterion(y_scores.view(-1, len(label_vocab)),
                             y.view(-1))  # requires tensors of shape (num-instances, num-labels) and (num-instances)

            # gather loss statistics
            total_loss += loss.item()
            num_loss += 1

            # gather accuracy statistics
            y_pred = torch.max(y_scores, 2)[1]  # compute highest-scoring tag
            #       mask = (y != 0) # ignore <pad> tags
            mask = (y != 0) * (y != label_vocab['UNK'])
            correct += torch.sum((y_pred == y) * mask)  # compute number of correct predictions
            num_perf += torch.sum(mask).item()

    return total_loss / num_loss, correct.item() / num_perf


# without training, accuracy should be a bit less than 2% (chance of getting a label correct)
# perf(rnn_model, valid_loader)

def fit_intermixing(model, epochs, train_loader, train_loader2, valid_loader, label_vocab, lr=1e-2, alphabet=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = num = 0
        for i, (data1, data2) in enumerate(zip(cycle(train_loader), train_loader2)):
            optimizer.zero_grad()  # start accumulating gradients
            x = data1[0].to(device)
            y = data1[1].to(device)

            if x.shape[1] >= 512:
                continue
            else:
                if alphabet:
                    y_scores = model(x, alphabet[i])
                else:
                    y_scores = model(x)
                loss = criterion(y_scores.view(-1, len(label_vocab)), y.view(-1))
                if not torch.isnan(loss):
                    loss.backward()  # compute gradients though computation graph
                    optimizer.step()  # modify model parameters
                    total_loss += loss.item()
                num += 1

            x = data2[0].to(device)
            y = data2[1].to(device)

            if x.shape[1] >= 512:
                continue
            else:
                if alphabet:
                    y_scores = model(x, alphabet[i])
                else:
                    y_scores = model(x)
                loss = criterion(y_scores.view(-1, len(label_vocab)), y.view(-1))
                if not torch.isnan(loss):
                    loss.backward()  # compute gradients though computation graph
                    optimizer.step()  # modify model parameters
                    total_loss += loss.item()
                num += 1
        print(1 + epoch, total_loss / num, *perf(model, valid_loader, label_vocab))



def fit_interleaving(model, epochs, train_loader, train_loader2, valid_loader, label_vocab, lr=1e-2, alphabet=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = num = 0
        for i, (data1, data2) in enumerate(zip(cycle(train_loader), train_loader2)):
            optimizer.zero_grad()  # start accumulating gradients
            x = data1[0].to(device)
            y = data1[1].to(device)
            # loss1 = loss2 = 0
            if x.shape[1] >= 512:
                continue
            else:
                if alphabet:
                    y_scores = model(x, alphabet[i])
                else:
                    y_scores = model(x)
                loss1 = criterion(y_scores.view(-1, len(label_vocab)), y.view(-1))
                # if not torch.isnan(loss):
                #     loss.backward()  # compute gradients though computation graph
                #     optimizer.step()  # modify model parameters
                #     total_loss += loss.item()
                num += 1

            x = data2[0].to(device)
            y = data2[1].to(device)

            if x.shape[1] >= 512:
                continue
            else:
                if alphabet:
                    y_scores = model(x, alphabet[i])
                else:
                    y_scores = model(x)
                loss2 = criterion(y_scores.view(-1, len(label_vocab)), y.view(-1))
                num += 1

            loss_total = 0
            if not torch.isnan(loss1):
                loss_total += loss1
            if not torch.isnan(loss2):
                loss_total += loss2

            if not torch.isnan(loss_total):

                loss_total.backward()  # compute gradients though computation graph
                optimizer.step()  # modify model parameters
                total_loss += loss_total.item()

        print(1 + epoch, total_loss / num, *perf(model, valid_loader, label_vocab))
