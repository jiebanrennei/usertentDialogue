import os
import pickle
import json
from sentence_transformers import SentenceTransformer
import torch
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from stanfordcorenlp import StanfordCoreNLP
import string

string.punctuation
def Syntactic(yic_content):
    nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2017-06-09', lang='zh')
os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.path.abspath(os.path.dirname(os.getcwd()))
os.path.abspath(os.path.join(os.getcwd(), ".."))
dataset ='mr'
input = os.sep.join(['..', 'data_tgcn', dataset, 'build_train', dataset])
output = os.sep.join(['..', 'data_tgcn', dataset, 'lstm'])

# with open(input + '.data.json', 'r') as file:
#     my_data = json.load(file)
#     print(my_data)

# with open(input + '.chinesefen1.txt', 'r',encoding='utf-8') as file:
#     lines = file.readlines()
# file.close()
# model_path = 'models/bert'
# bi_encoder = SentenceTransformer(model_path)
# candidates = []
# s_contentemb_list = []
# with open('../data_tgcn/mr/build_train/testing_new', 'rb') as fp:
#     testing = pickle.load(fp)
# for i in range(len(testing)):
#     candidates.append(testing[i][0])
# print(candidates)
# for candidate in candidates:
#     emd = bi_encoder.encode(candidate, convert_to_tensor=True, device='cuda')
#     s_contentemb_list.append(emd)
# print(s_contentemb_list)
# output2 = open(output + '/{}_semb1.pkl'.format(dataset), 'wb')
# pickle.dump(s_contentemb_list, output2)

# with open(input + '_emb.pkl', 'wb') as f:
#     list = []
#     for line in lines:
#         emb = {}
#         windows = line.strip("\n").split(" ")
#         for i in range(len(windows)):
#             emd = bi_encoder.encode(windows[i], convert_to_tensor=True, device='cuda')
#             for synset in wn.synsets(windows[i], lang='cmn'):
#                 if len(synset.lemma_names('cmn')) != 0:
#                     for lemma in synset.lemma_names('cmn'):
#                         lemmaenb = bi_encoder.encode(lemma, convert_to_tensor=True, device='cuda')
#                         stacked_tensor = torch.stack((emd, lemmaenb))
#                         emd = torch.mean(stacked_tensor, dim=0)
#             emb[windows[i]] = emd
#         list.append(emb)
#     pickle.dump(list,f)
#
#
# with open(input+'_emb.pkl', "rb") as file:
#     sentenceemb = pickle.load(file)
# with open("../data_tgcn/mr/lstm/mr_emb.pkl", "rb") as file:
#     wordemb = pickle.load(file)
# print(wordemb)
q = []
c = []
training = pd.read_csv('../data/training_ct1.csv', sep='\t')
def contruct_training_instance(x, row_index):
    print(row_index,x['text1'], x['text2'])
    q.append(x['text1'])
    c.append(x['text2'])
training.apply(lambda x: contruct_training_instance(x, x.name), axis=1)
print(q)
print(c)
file = open(input + '.tqchinesefen.txt', "w",encoding="utf-8")
nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2017-06-09', lang='zh')
for doc_id in range(len(q)):
    # print(doc_id)
    words = q[doc_id]
    words = words.split("\n")
    rela = []
    for window in words:
        if window == ' ':
            continue
        punctuation_string = string.punctuation
        for i in punctuation_string:
            window = window.replace(i, '')
        # print(window)
        res = nlp.dependency_parse(window)
        # print('Tokenize:', nlp.word_tokenize(window))
        file.write(' '.join(nlp.word_tokenize(window)))
        file.write("\n")
        fen = nlp.word_tokenize(window)
file.close()
nlp.close()

file = open(input + '.tcchinesefen.txt', "w",encoding="utf-8")
nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2017-06-09', lang='zh')
for doc_id in range(len(c)):
    # print(doc_id)
    words = c[doc_id]
    words = words.split("\n")
    rela = []
    for window in words:
        if window == ' ':
            continue
        punctuation_string = string.punctuation
        for i in punctuation_string:
            window = window.replace(i, '')
        # print(window)
        res = nlp.dependency_parse(window)
        # print('Tokenize:', nlp.word_tokenize(window))
        file.write(' '.join(nlp.word_tokenize(window)))
        file.write("\n")
        fen = nlp.word_tokenize(window)
file.close()
nlp.close()
tq = []
with open(input + '.tqdata.json', 'w') as f:
    nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2017-06-09', lang='zh')
    for doc_id in range(len(q)):
        # print(doc_id)
        words = q[doc_id]
        words = words.split("\n")
        rela = []
        for window in words:
            if window == ' ':
                continue
            punctuation_string = string.punctuation
            for i in punctuation_string:
                window = window.replace(i, '')
            # print(window)
            res = nlp.dependency_parse(window)
            tq.append(res)
    nlp.close()
    json.dump(tq, f)
tc = []
# res = nlp.dependency_parse(window)
with open(input + '.tcdata.json', 'w') as f1:
    nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2017-06-09', lang='zh')
    for doc_id in range(len(c)):
        # print(doc_id)
        words = c[doc_id]
        words = words.split("\n")
        rela = []
        for window in words:
            if window == ' ':
                continue
            punctuation_string = string.punctuation
            for i in punctuation_string:
                window = window.replace(i, '')
            # print(window)
            res = nlp.dependency_parse(window)
            tc.append(res)
    nlp.close()
    json.dump(tc, f1)