from gcn import GCN
import torch
import os
import pickle
from gcn1 import GCN1
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer

def trainC(location1,location2):
    dataset = 'mr'
    os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.path.abspath(os.path.dirname(os.getcwd()))
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    input1 = os.sep.join(['..', 'data_tgcn', dataset, 'build_train', dataset])

    with open(location1, "rb") as file:
        awordemb = pickle.load(file)
    with open(location2, "rb") as file:
        asentenceemb = pickle.load(file)
    with open("../data_tgcn/mr/lstm/mr_emb.pkl", "rb") as file:
        wordemb = pickle.load(file)
    with open("../data_tgcn/mr/lstm/mr_semb.pkl", "rb") as file:
        sentenceemb = pickle.load(file)
    with open("../data_tgcn/mr/build_train/ind.mr.adjC", "rb") as file:
        adjC = pickle.load(file)
    with open("../data_tgcn/mr/build_train/ind.mr.adjC1", "rb") as file:
        adjC1 = pickle.load(file)
    with open("../data_tgcn/mr/build_train/ind.mr.adjC2", "rb") as file:
        adjC2 = pickle.load(file)
    vocab = []
    f = open(input1 + '_chinesevocab.txt', 'r', encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        vocab.append(line.strip())
    f.close()

    nonzero_indices = adjC.nonzero()
    nonzero_values = adjC.data

    if len(sentenceemb) == 768:
        a = 1
    else:
        a = len(sentenceemb)
    adjC = torch.zeros(len(vocab) + len(asentenceemb) + a, len(vocab) + len(asentenceemb) + a, dtype=torch.float)

    for i in range(len(nonzero_indices[0])):
        adjC[nonzero_indices[0][i], nonzero_indices[1][i]] = nonzero_values[i]

    nonzero_indices1 = adjC1.nonzero()
    nonzero_values1 = adjC1.data

    adjC1 = torch.zeros(len(vocab) + len(asentenceemb) + a, len(vocab) + len(asentenceemb) + a, dtype=torch.float)

    for i in range(len(nonzero_indices1[0])):
        adjC1[nonzero_indices1[0][i], nonzero_indices1[1][i]] = nonzero_values1[i]

    nonzero_indices2 = adjC2.nonzero()
    nonzero_values2 = adjC2.data

    adjC2 = torch.zeros(len(vocab) + len(asentenceemb) + a, len(vocab) + len(asentenceemb) + a, dtype=torch.float)

    for i in range(len(nonzero_indices2[0])):
        adjC2[nonzero_indices2[0][i], nonzero_indices2[1][i]] = nonzero_values2[i]

    node_features = torch.zeros(len(vocab) + len(asentenceemb) + a, 768, dtype=torch.float)
    for i in range(len(vocab) + len(asentenceemb) + a):
        if i < len(awordemb):
            node_features[i] = awordemb[vocab[i]]
        if i >= len(awordemb) and i <len(vocab):
            if vocab[i] in awordemb:
                node_features[i] = awordemb[vocab[i]]
            else:
                node_features[i] = wordemb[vocab[i]]
        if i>= len(vocab) and i < (len(vocab) + len(asentenceemb)):
            node_features[i] = asentenceemb[i - len(vocab)]
        if i >= (len(vocab) + len(asentenceemb)):
            node_features[i] = sentenceemb[i - len(vocab) - len(asentenceemb)]

    x = node_features

    input_dim = 768
    hidden_dim = 1536
    output_dim = 768
    model = GCN(input_dim, hidden_dim, output_dim)
    model = model.cuda()
    x = x.cuda()
    adjC = adjC.cuda()
    adjC1 = adjC1.cuda()
    adjC2 = adjC2.cuda()

    pim,syntactic,semantic= model(x, adjC, adjC1, adjC2, len(vocab))
    size = 3
    adj = torch.ones(size, size)
    adj.fill_diagonal_(0)
    adj = adj.to('cuda')
    model1 = GCN1(input_dim, hidden_dim, output_dim)
    model1 = model1.to('cuda')
    output = torch.zeros(len(pim), 768, dtype=torch.float)
    output = output.cuda()
    for i in range(len(pim)):
        features1 = torch.zeros(3,768, dtype=torch.float)
        features1 = features1.cuda()
        features1[0] = pim[i]
        features1[1] = syntactic[i]
        features1[2] = semantic[i]
        output[i] = model1(features1,adj)
    return output