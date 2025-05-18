import pickle

candidates_path = '../data/candidates'
with open(candidates_path, 'rb') as fp:
    candidates1 = pickle.load(fp)
for item in candidates1:
    print(item)