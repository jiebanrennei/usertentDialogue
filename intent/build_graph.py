import os
import pickle
import pickle as pkl
import random
from math import log
import numpy as np
import scipy.sparse as sp
import pickle

def build_graph(location1,location2,location3):
    # build corpus
    dataset = 'mr'
    os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.path.abspath(os.path.dirname(os.getcwd()))
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    input1 = os.sep.join(['..', 'data_tgcn', dataset, 'build_train', dataset])
    # print(input1)'..\\data_tgcn\\mr\\build_train\\mr'
    input2 = os.sep.join(['..', 'data_tgcn', dataset, 'stanford'])
    # print(input2)'..\\data_tgcn\\mr\\stanford'
    input3 = os.sep.join(['..', 'data_tgcn', dataset, 'lstm'])
    # print(input3)'..\\data_tgcn\\mr\\lstm'
    input4 = os.sep.join(['..', 'data_tgcn', dataset, 'lstm', dataset])
    # print(input4)'..\\data_tgcn\\mr\\lstm\\mr'
    output1 = os.sep.join(['..', 'data_tgcn', dataset, 'build_train', dataset])
    # print(output1)'..\\data_tgcn\\mr\\build_train\\mr'
    output2 = os.sep.join(['..', 'data_tgcn', dataset, 'build_train'])
    # print(output2)'..\\data_tgcn\\mr\\build_train'

    window_size = 7
    # yic_content_list = []
    # f = open(input1 + location1, 'r', encoding="utf-8")
    # lines = f.readlines()
    # for line in lines:
    #     yic_content_list.append(line.strip())
    # f.close()
    #
    # f = open(input1 + '.chinesefen.txt', 'r', encoding="utf-8")
    # lines = f.readlines()
    # for line in lines:
    #     yic_content_list.append(line.strip())
    # f.close()

    doc_content_list = []
    f = open(input1 + location1, 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()

    f = open(input1 + '.chinesefen.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()
    # print(doc_content_list)

    # build vocab
    word_freq = {}  # Create an empty dictionary to store words and their occurrence frequencies.
    word_set = []  # Create an empty set to store all the words that have appeared, so as to build a vocabulary.
    for doc_words in doc_content_list:
        words = doc_words.split()
        for word in words:
            if word not in word_set:
                word_set.append(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = word_set
    vocab_size = len(vocab)
    # print(vocab)
    # print(word_freq)

    word_doc_list = {}  # It is used to store in which documents each word appears.
    # It is used to construct a dictionary, where each key is a word and the corresponding value is a list containing the documents in which the word appears.
    for i in range(len(doc_content_list)):
        doc_words = doc_content_list[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)
    # print(word_doc_list)
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)
    # print(word_doc_freq)

    word_id_map = {}  # Store the mapping relationship from words to IDs.
    id_word_map = {}  # Store the mapping relationship from IDs to words.
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i
        id_word_map[i] = vocab[i]
    # print(word_id_map)
    # print(id_word_map)
    vocab_str = '\n'.join(vocab)

    f = open(output1 + '_chinesevocab.txt', 'w',encoding='utf-8')
    f.write(vocab_str)
    f.close()

    windows = []

    for doc_words in doc_content_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)

    word_window_freq = {}  # Store the frequency statistics of words appearing in the window.
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_in_total = {}

    # The frequency statistics of each word pair appearing in the window are stored in the `word_pair_count` dictionary.
    # The key of this dictionary is a string representation composed of two word IDs, and the corresponding value is the number of times the word pair appears in the window.
    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
    # print(word_pair_count)

    row = []
    col = []
    weight = []
    weight1 = []
    weight2 = []

    # Construct edge weights based on the Stanford syntactic dependency.
    data1 = pickle.load(open(input2 + "/{}_chsy.pkl".format(dataset), "rb"))
    data11 = pickle.load(open(input2 + location2.format(dataset), "rb"))


    # Construct edge weights based on semantic dependency.
    data2 = pickle.load(open(input4 + "_chse.pkl", "rb"))
    data21 = pickle.load(open(input4 + location3, "rb"))

    # word_pair_in_doc = {}

    # compute weights
    num_window = len(windows)
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        # pmi
        row.append(i)
        col.append(j)
        weight.append(pmi)
        # Syntactic Dependency
        if i not in id_word_map or j not in id_word_map:
            continue
        newkey = id_word_map[i] + ',' + id_word_map[j]

        # for doc_id in range(len(doc_content_list)):
        #     words = doc_content_list[doc_id]
        #     words = words.split("\n")
        #     for window in words:
        #         if id_word_map[i] in window and id_word_map[j] in window:
        #             if newkey in word_pair_in_doc:
        #                 word_pair_in_doc[newkey] += 1
        #             else:
        #                 word_pair_in_doc[newkey] = 1
        wei = 0
        if newkey in data1:
            wei += data1[newkey]
        if newkey in data11:
            wei += data11[newkey]
        weight1.append(wei/count)

        # Semantic Dependency
        wei = 0
        if newkey in data2:
            wei += data2[newkey]
        if newkey in data21:
            wei += data21[newkey]
        weight2.append(wei/count)

    # doc word frequency
    weight_tfidf = []  # Weight values based on TF-IDF (Term Frequency-Inverse Document Frequency)
    doc_word_freq = {}
    for doc_id in range(len(doc_content_list)):
        doc_words = doc_content_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(doc_content_list)):
        doc_words = doc_content_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            row.append(i + vocab_size)
            col.append(j)
            idf = log(1.0 * len(doc_content_list) /
                      word_doc_freq[vocab[j]])
            weight_tfidf.append(freq * idf)
            doc_word_set.add(word)

    weight = weight + weight_tfidf
    node_size = vocab_size + len(doc_content_list)
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))
    # print(adj)
    f = open(output2 + '/ind.{}.adjC'.format(dataset), 'wb')
    pkl.dump(adj, f)
    f.close()

    weight = weight1 + weight_tfidf
    node_size = vocab_size + len(doc_content_list)
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))
    # print(adj)
    f = open(output2 + '/ind.{}.adjC1'.format(dataset), 'wb')
    pkl.dump(adj, f)
    f.close()

    weight = weight2 + weight_tfidf
    node_size = vocab_size + len(doc_content_list)
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))
    f = open(output2 + '/ind.{}.adjC2'.format(dataset), 'wb')
    pkl.dump(adj, f)
    f.close()