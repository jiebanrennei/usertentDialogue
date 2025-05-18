import os
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
import torch

import pickle
def Semantic(yic_content,location):
    model_path = 'models/bert'
    bi_encoder = SentenceTransformer(model_path)
    os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.path.abspath(os.path.dirname(os.getcwd()))
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    dataset = 'mr'
    input = os.sep.join(['..', 'data_tgcn', dataset, 'build_train', dataset])
    # output = os.sep.join(['..', 'data_tgcn', dataset, 'stanford'])
    output = os.sep.join(['..', 'data_tgcn', dataset, 'lstm'])
    # print(input3)'..\\data_tgcn\\mr\\lstm'

    f = open(output + location.format(dataset), 'rb')
    data = pickle.load(f)
    f.close()
    stop_words = set()
    with open(input + '.stop.txt', 'r', encoding="utf-8") as f:
        for line in f:
            stop_words.add(line.strip())

    rela_pair_count_str = {}
    yic_content_list = []
    f = open(input + '.chinesefen.txt', 'r', encoding="utf-8")
    # f = open(input + '.candidates_fen.txt', 'r', encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        yic_content_list.append(line.strip())
    f.close()
    emb = {}
    for doc_id in range(len(yic_content_list)):
        words = yic_content_list[doc_id]
        words = words.split("\n")
        for window in words:
            if window == ' ':
                continue
            windows = window.split()
            for i in range(len(windows)):
                if (windows[i] in emb):
                    continue
                if (windows[i] in data):
                    continue
                emd = bi_encoder.encode(windows[i], convert_to_tensor=True, device='cuda')
                # for synset in wn.synsets(windows[i], lang='cmn'):
                #     if len(synset.lemma_names('cmn')) != 0:
                #         for lemma in synset.lemma_names('cmn'):
                #             lemmaenb = bi_encoder.encode(lemma, convert_to_tensor=True, device='cuda')
                #             stacked_tensor = torch.stack((emd, lemmaenb))
                #             emd = torch.mean(stacked_tensor, dim=0)
                emb[windows[i]] = emd
    output2=open(output + '/{}_emb.pkl'.format(dataset),'wb')
    # output2 = open(output + '/{}_candidatesemb.pkl'.format(dataset), 'wb')
    pickle.dump(emb, output2)
    for doc_id in range(len(yic_content_list)):
        words = yic_content_list[doc_id]
        words = words.split("\n")
        for window in words:
            if window == ' ':
                continue
            windows = window.split()
            for num1 in range(len(windows)):
                if windows[num1] in stop_words:
                    continue
                for num2 in range(num1 + 1, len(windows)):
                    if windows[num2] in stop_words:
                        continue
                    sim = torch.cosine_similarity(emb[windows[num1]], emb[windows[num2]], dim=0)
                    if sim > 0.5:
                        word_pair_str = windows[num1] + ',' + windows[num2]
                        if word_pair_str in rela_pair_count_str:
                            rela_pair_count_str[word_pair_str] += 1
                        else:
                            rela_pair_count_str[word_pair_str] = 1
                        # two orders
                        word_pair_str = windows[num2] + ',' + windows[num1]
                        if word_pair_str in rela_pair_count_str:
                            rela_pair_count_str[word_pair_str] += 1
                        else:
                            rela_pair_count_str[word_pair_str] = 1
    # print(rela_pair_count_str)
    output1=open(output + '/{}_chse.pkl'.format(dataset),'wb')
    # output1 = open(output + '/{}_candidates.pkl'.format(dataset), 'wb')
    pickle.dump(rela_pair_count_str, output1)

    s_content_list = yic_content
    # f = open(input + '.chinese.txt', 'r', encoding="utf-8")
    # lines = f.readlines()
    # for line in lines:
    #     s_content_list.append(line.strip())
    # f.close()
    s_contentemb_list = []
    for doc_id in range(len(s_content_list)):
        words = s_content_list[doc_id]
        words = words.split("\n")
        for window in words:
            if window == ' ':
                continue
            emd = bi_encoder.encode(window, convert_to_tensor=True, device='cuda')
            s_contentemb_list.append(emd)
    # print(len(s_contentemb_list))
    output2=open(output + '/{}_semb.pkl'.format(dataset),'wb')
    # output2 = open(output + '/{}_candidatessemb.pkl'.format(dataset), 'wb')
    pickle.dump(s_contentemb_list, output2)

# candidates = []
# candidates_path = '../data_tgcn/mr/build_train/candidates1.txt'
# f = open(candidates_path, 'r', encoding="utf-8")
# lines = f.readlines()
# for line in lines:
#     candidates.append(line.strip())
# f.close()
# Semantic(candidates,'1111')
