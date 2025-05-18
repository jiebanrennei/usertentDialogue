import os
import pickle
import string
import json

string.punctuation
def Syntactic1(location1,location2,index):

    os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.path.abspath(os.path.dirname(os.getcwd()))
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    dataset ='mr'
    input = os.sep.join(['..', 'data_tgcn', dataset, 'build_train', dataset])
    output = os.sep.join(['..', 'data_tgcn', dataset, 'stanford'])

    stop_words = set()
    with open(input + '.stop.txt', 'r', encoding="utf-8") as f:
        for line in f:
            stop_words.add(line.strip())

    with open(input + location2, 'r') as file:
        my_data = json.load(file)

    with open(input + location1, 'r', encoding='utf-8') as file:
        line = file.readline()
        counts = 1
        while line:
            if counts >= index + 1:
                break
            line = file.readline()
            counts += 1
    fen = line.strip("\n").split(" ")

    file = open(input + '.chinesefen.txt', "w", encoding="utf-8")
    file.write(line.strip("\n"))
    file.write("\n")

    rela_pair_count_str = {}
    rela = my_data[index]
    for pair in rela:
        if pair[0] == 'ROOT' or pair[1] == 'ROOT':
            continue
        if fen[pair[1] - 1] == fen[pair[2] - 1]:
            continue
        if fen[pair[1] - 1] in string.punctuation or fen[pair[2] - 1] in string.punctuation:
            continue
        if fen[pair[1] - 1] in stop_words or fen[pair[2] - 1] in stop_words:
            continue
        word_pair_str = fen[pair[1] - 1] + ',' + fen[pair[2] - 1]
        if word_pair_str in rela_pair_count_str:
            rela_pair_count_str[word_pair_str] += 1
        else:
            rela_pair_count_str[word_pair_str] = 1
        # two orders
        word_pair_str = fen[pair[2] - 1] + ',' + fen[pair[1] - 1]
        if word_pair_str in rela_pair_count_str:
            rela_pair_count_str[word_pair_str] += 1
        else:
            rela_pair_count_str[word_pair_str] = 1

    # print(rela_pair_count_str)
    output1=open(output + '/{}_chsy.pkl'.format(dataset),'wb')
    # output1 = open(output + '/{}_candidates.pkl'.format(dataset), 'wb')
    pickle.dump(rela_pair_count_str, output1)


# candidates = []
# with open('../data_tgcn/mr/build_train/testing_new', 'rb') as fp:
#     testing = pickle.load(fp)
# for i in range(len(testing)):
#     candidates.append(testing[i][0])
# print(len(candidates))
# Syntactic1(4)

