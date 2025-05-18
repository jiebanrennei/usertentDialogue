from sentence_transformers import SentenceTransformer, util
import pickle
import torch
import torch.nn.functional as F
import random
import Syntactic
import Syntactic1
import Semantic
import Semantic1
import build_graph
import trainC
import statistics

class Candidate_Ranker(object):
    
    def __init__(self,model_path = 'models/bert'):
        self.bi_encoder = SentenceTransformer(model_path)
        self.candidates = []
        self.candidates_path = '../data_tgcn/mr/build_train/candidates1.txt'
        f = open(self.candidates_path, 'r', encoding="utf-8")
        lines = f.readlines()
        for line in lines:
            self.candidates.append(line.strip())
        f.close()

        self.candidates_path = '../data/candidates'
        with open (self.candidates_path, 'rb') as fp:
                self.candidates1 = pickle.load(fp)

        self.candidates_path = '../data/candidates1'
        with open (self.candidates_path, 'rb') as fp:
                self.candidates2 = pickle.load(fp)

        self.querise = []
        self.querise_path = '../data_tgcn/mr/build_train/dialogues1.txt'
        f = open(self.querise_path, 'r', encoding="utf-8")
        lines = f.readlines()
        for line in lines:
            self.querise.append(line.strip())
        f.close()

        model_path = 'models/bert'
        self.bi_encoder = SentenceTransformer(model_path) #, device = 'cpu'
        self.candidates_embeddings = self.bi_encoder.encode(self.candidates1, convert_to_tensor=True, device='cuda')
    def retrieve_top_k(self,emb, embs, k = 10, metric='cosine'):
        """Retrieve the top k candidate nodes that are most similar to the dialogue node."""
        if metric == 'cosine':
            similarities = F.cosine_similarity(emb,embs, dim=-1)
        elif metric == 'euclidean':
            distances = torch.cdist(emb.unsqueeze(dim=0),embs, p=2)
            similarities = torch.abs(distances.squeeze())
        else:
            raise ValueError('Unsupported similarity metric')

        scores, indices = torch.topk(similarities, k)
        results = [indices,scores]
        return results
    # This function will search all standard intents for dialogue that match the dialogue
    def search_candidates(self, dialogue, top_k = 5, return_scores= False, threshold = 0):
        question_embedding = self.bi_encoder.encode(dialogue, convert_to_tensor=True, device ='cuda')
        hits = util.semantic_search(question_embedding, self.candidates_embeddings, top_k = top_k)
        hits = hits[0]
        if return_scores:
            hits = [(self.candidates1[hit['corpus_id']],hit['score']) for hit in hits if hit['score']>threshold]
        else:
            hits = [self.candidates1[hit['corpus_id']] for hit in hits if hit['score']>threshold]
        return hits
    
    def topk(self,ls, k):
        indices = range(len(ls))
        indices = sorted(indices, key = lambda x: ls[x], reverse = True)
        return indices[:k]

    def search_hard_negatives(self, anchor = 'dialogue', dialogue = None, true_candidate = None, sampling_method = 'topK', positive_threshold = 0.5, beta = 1,num_negatives = 1,index = 0):
        """Finds hard negative candidates to be used for training."""
        a = []
        b = []
        score = 0
        score_true = self.cos_sim(dialogue, true_candidate)[0]
        if anchor == 'dialogue':
           candidates = []
           candidates.append(dialogue)
           # candidates.append(true_candidate)
           location1 = '.tqchinesefen.txt'
           location2 = '.tqdata.json'
           Syntactic1.Syntactic1(location1,location2,index)
           location = '/{}_candidates.pkl'
           Semantic.Semantic(candidates,location)
           location1 = '.candidates_fen.txt'
           location2 = '/{}_candidates.pkl'
           location3 = '_candidates.pkl'
           build_graph.build_graph(location1,location2,location3)
           location1 = "../data_tgcn/mr/lstm/mr_candidatesemb.pkl"
           location2 = "../data_tgcn/mr/lstm/mr_candidatessemb.pkl"
           emb = trainC.trainC(location1, location2)
           dialogue_embeddings = emb[len(emb)-1]
           # true_candidate_dialogue_embeddings = emb[len(emb)-1]
           a = dialogue_embeddings
           # b = true_candidate_dialogue_embeddings
           candidates_embeddings = emb[:len(emb)-1, :]
           hits = util.semantic_search(dialogue_embeddings, candidates_embeddings, top_k=len(self.candidates))
           hits = hits[0]
           # hits = name
           hits = [(self.candidates[hit['corpus_id']],min(max(hit['score'],0.001),1),hit['corpus_id']) for hit in hits if self.candidates[hit['corpus_id']]!=true_candidate]
           # hits_cp = [item for item in hits if item[1] >= positive_threshold]  # positive predictions
           hits_cp = [item for item in hits if item[1]>=positive_threshold and item[1] >= score_true - 0.15 and item[1] <= 0.15] # positive predictions
        elif anchor == 'candidate':
           querise = []
           querise.append(true_candidate)
           # querise.append(dialogue)
           location1 = '.tcchinesefen.txt'
           location2 = '.tcdata.json'
           Syntactic1.Syntactic1(location1, location2, index)
           location = '/{}_querise.pkl'
           Semantic.Semantic(querise,location)
           location1 = '.querise_fen.txt'
           location2 = '/{}_querise.pkl'
           location3 = '_querise.pkl'
           build_graph.build_graph(location1,location2,location3)
           location1 = "../data_tgcn/mr/lstm/mr_queriseemb.pkl"
           location2 = "../data_tgcn/mr/lstm/mr_querisesemb.pkl"
           emb = trainC.trainC(location1,location2)
           candidate_embedding = emb[len(emb) - 1]
           # querise_embedding = emb[len(emb) - 1]
           a = candidate_embedding
           # b = querise_embedding
           queries_embeddings = emb[:len(emb) - 1, :]
           hits = util.semantic_search(candidate_embedding, queries_embeddings, top_k=len(self.querise))
           hits = hits[0]



           hits = [(self.querise[hit['corpus_id']], min(max(hit['score'], 0.001), 1)) for hit in hits if
                   self.querise[hit['corpus_id']] != dialogue]
           # hits_cp = [item for item in hits if item[1] >= positive_threshold]  # positive predictions
           hits_cp = [item for item in hits if item[1] >= positive_threshold and item[1] >= score_true - 0.15 and item[1] <= 0.15]  # positive predictions bywyf
        else:
           return None

        probs = [item[1] for item in hits_cp]
        candts = [item[0] for item in hits_cp]

        if sampling_method == 'topK':
            while len(hits_cp) <= num_negatives:
                hits_cp.append(random.sample(hits, 1)[0])
            probs = [item[1] for item in hits_cp]
            candts = [item[0] for item in hits_cp]
            indices = self.topk(probs, num_negatives)

        elif sampling_method == 'topK_with_E-FN':
           hits_cp = [hit for hit in hits_cp if hit[1]<=beta]
           while len(hits_cp)<=num_negatives:
                   hits_cp.append(random.sample(hits,1)[0])
           probs = [item[1] for item in hits_cp]
           candts = [item[0] for item in hits_cp]
           indices = self.topk(probs, num_negatives)

        elif sampling_method == 'larger_than_true':
             # score_true = torch.cosine_similarity(a, b, dim=0)
             score_true = self.cos_sim(dialogue, true_candidate)[0]
             hits_cp = [hit for hit in hits_cp if hit[1]>=score_true]
             while len(hits_cp)<=num_negatives:
                   hits_cp.append(random.sample(hits,1)[0])
             probs = [item[1] for item in hits_cp]
             candts = [item[0] for item in hits_cp]
             indices = self.topk(probs, num_negatives)

        elif sampling_method == 'larger_than_true_with_E-FN':
             # score_true = torch.cosine_similarity(a, b, dim=0)
             score_true = self.cos_sim(dialogue, true_candidate)[0]
             hits_cp = [hit for hit in hits_cp if hit[1]>=score_true and hit[1]<=beta]
             while len(hits_cp)<=num_negatives:
                   hits_cp.append(random.sample(hits,1)[0])
             probs = [item[1] for item in hits_cp]
             candts = [item[0] for item in hits_cp]
             indices = self.topk(probs, num_negatives)

        elif sampling_method == 'multinomial1':
           res = []
           hits_cp1 = [item for item in hits if item[1] >= (score_true + 0.15)]
           probs = [item[1] for item in hits_cp1]
           candts = [item[0] for item in hits_cp1]
           probs = torch.tensor(probs, dtype=torch.float)
           if(len(hits_cp1) < num_negatives):
               if(len(hits_cp1) > 0):
                   res.extend([candts[i] for i in torch.multinomial(probs, len(hits_cp1), replacement=False)])
                   hits_cp2 = [item for item in hits if item[1] >= score_true - 0.15 and item[1] < score_true - 0.15]
                   while len(hits_cp2) < num_negatives-len(hits_cp1):
                       hits_cp2.append(random.sample(hits, 1)[0])
                   probs = [item[1] for item in hits_cp2]
                   candts = [item[0] for item in hits_cp2]
                   probs = torch.tensor(probs, dtype=torch.float)
                   res.extend([candts[i] for i in torch.multinomial(probs, num_negatives-len(hits_cp1), replacement=False)])
               else:
                   while len(hits_cp1) < num_negatives:
                       hits_cp1.append(random.sample(hits, 1)[0])
                   probs = [item[1] for item in hits_cp1]
                   candts = [item[0] for item in hits_cp1]
                   probs = torch.tensor(probs, dtype=torch.float)
                   res.extend([candts[i] for i in torch.multinomial(probs, len(hits_cp1), replacement=False)])
           else:
               res.extend([candts[i] for i in torch.multinomial(probs, num_negatives, replacement=False)])
           return res

        elif sampling_method == 'multinomial':
           while len(hits_cp)<=num_negatives:
                   hits_cp.append(random.sample(hits,1)[0])

           probs = [item[1] for item in hits_cp]
           candts = [item[0] for item in hits_cp]
           probs = torch.tensor(probs, dtype=torch.float)
           indices = torch.multinomial(probs, num_negatives, replacement=False)

        elif sampling_method == 'multinomial_with_E-FN':
           hits_cp = [hit for hit in hits_cp if hit[1]<=beta]
           while len(hits_cp)<=num_negatives:
                   hits_cp.append(random.sample(hits,1)[0])
           probs = [item[1] for item in hits_cp]
           candts = [item[0] for item in hits_cp]
           probs = torch.tensor(probs, dtype=torch.float)
           indices = torch.multinomial(probs, num_negatives, replacement=False)

        else:
           indices = random.sample(range(len(candts)), k = min(num_negatives,len(candts)))

        return [candts[i] for i in indices]
    def cos_sim(self, dialogue, target):
        dialogue_emb = self.bi_encoder.encode(dialogue, convert_to_tensor=True, device ='cuda')
        target_emb = self.bi_encoder.encode(target, convert_to_tensor=True, device ='cuda')
        return util.cos_sim(dialogue_emb,target_emb).tolist()[0]