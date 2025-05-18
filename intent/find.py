import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import pandas as pd
import Evaluation
import math
import pickle
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from BidirectionalHardNegativesRankingLoss import BidirectionalHardNegativesRankingLoss
from candidate_ranker import Candidate_Ranker
pd.set_option('mode.chained_assignment', None)

dataset = 'mr'
os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.path.abspath(os.path.dirname(os.getcwd()))
os.path.abspath(os.path.join(os.getcwd(), ".."))
input1 = os.sep.join(['..', 'data_tgcn', dataset, 'build_train', dataset])

count = 0
def increment():
    global count
    count += 1
    return count

def preprocessing_dataset(cls, positive_threshold, beta, training, validation, train_batch_size,
                          num_hard_negative_queries, num_hard_negative_candidates, sampling_method):
    logging.info("Pre-processing training/validation dataset: ")
    # Define a function to generate (positive, hard negative) pairs
    def bi_encoder(index,dialogue, true_candidate):
        predictions_candidate = cls.search_hard_negatives(anchor='dialogue', dialogue=dialogue, true_candidate=true_candidate,
                                                          sampling_method=sampling_method,
                                                          positive_threshold=positive_threshold, beta=beta,
                                                          # num_negatives=num_hard_negative_queries,
                                                          num_negatives=num_hard_negative_candidates,
                                                          index = index)
        predictions_dialogue = cls.search_hard_negatives(anchor='candidate', dialogue=dialogue, true_candidate=true_candidate,
                                                      sampling_method=sampling_method,
                                                      positive_threshold=positive_threshold, beta=beta,
                                                      num_negatives=num_hard_negative_queries,
                                                      # num_negatives=num_hard_negative_candidates,
                                                      index = index)
        count = increment()
        print(count)
        return (predictions_candidate, predictions_dialogue)
    def contruct_training_instance(x, row_index):
        hard_negatives_intent, hard_negatives_dialogue = bi_encoder(row_index,x['text1'], x['text2'])
        return InputExample(texts=[x['text1'], *hard_negatives_dialogue, x['text2'], *hard_negatives_intent],
                            label=float(x['label']))

    def contruct_instance(x):
        return InputExample(texts=[x['text1'], x['text2']], label=float(x['label']))

    training['training_instances'] = training.apply(lambda x: contruct_training_instance(x, x.name), axis=1)
    train_examples = training['training_instances'].tolist()
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)

    validation['validation_instances'] = validation.apply(contruct_instance, axis=1)
    validation_examples = validation['validation_instances'].tolist()
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(validation_examples, name='intent-dev')
    return (train_dataloader, evaluator)

def build_model(model_path, max_seq_length):
    logging.info("Build model from: " + model_path)
    word_embedding_model = models.Transformer(model_path,
                                              max_seq_length=max_seq_length)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def train_func(model, model_path, train_dataloader, evaluator, num_hard_negatives):
    logging.info("Training: ")

    train_loss = BidirectionalHardNegativesRankingLoss(model=model, num_hard_negatives_dialogue=num_hard_negatives)
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=1,
              # evaluation_steps = 3,
              evaluation_steps=math.ceil(len(train_dataloader) * 0.25),
              warmup_steps=math.ceil(len(train_dataloader) * 0.1),
              output_path=model_path,
              show_progress_bar=True)

if __name__ == '__main__':
    training = pd.read_csv('../data/training_ct4.csv', sep='\t')
    validation = pd.read_csv('../data/validation.csv', sep='\t')

    con = []
    loc = "../data_tgcn/mr/lstm/mr_candidatessemb.pkl"
    candidates_path = '..\data_tgcn\mr\\build_train\\candidates1.txt'
    f = open(candidates_path, 'r', encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        con.append(line.strip())
    f.close()

    queries = []
    loc1 = "../data_tgcn/mr/lstm/mr_querisesemb.pkl"
    queries_path = '../data_tgcn/mr/build_train/dialogues.txt'
    f = open(queries_path, 'r', encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        queries.append(line.strip())
    f.close()

    train_batch_size = 3
    epochs = 8
    model_path = 'models/bert'
    for epoch in range(epochs):
        count = 0
        print("epoch:",epoch)
        max_seq_length = 32
        cls = Candidate_Ranker(model_path = model_path)
        positive_threshold = 0.5  # It is used to determine the threshold of positive samples.
        beta = 0.9997
        num_hard_negative_queries = 10  # The number of hard negative samples corresponding to each positive sample.
        num_hard_negative_candidates = 3
        sampling_method = 'multinomial'

        # start_row = 0
        # end_row = 1000
        # for i in range(int(len(training) / 1000)):

        s_contentemb = []
        for doc_id in range(len(con)):
            words = con[doc_id]
            words = words.split("\n")
            for window in words:
                if window == ' ':
                    continue
                emd = cls.bi_encoder.encode(window, convert_to_tensor=True, device='cuda')
                s_contentemb.append(emd)
        output2 = open(loc, 'wb')
        pickle.dump(s_contentemb, output2)
        s_contentemb = []

        q_contentemb = []
        for doc_id in range(len(queries)):
            words = queries[doc_id]
            words = words.split("\n")
            for window in words:
                if window == ' ':
                    continue
                emd = cls.bi_encoder.encode(window, convert_to_tensor=True, device='cuda')
                q_contentemb.append(emd)
        output2 = open(loc1, 'wb')
        pickle.dump(q_contentemb, output2)
        q_contentemb = []

        # train_batch = pd.DataFrame([])
        # train_batch = training.iloc[start_row:end_row]
        # start_row += 1000
        # end_row += 1000
        # Generating hard negatives for training data
        train_dataloader, evaluator = preprocessing_dataset(cls, positive_threshold, beta, training, validation,
                                                        train_batch_size, num_hard_negative_queries,
                                                        num_hard_negative_candidates, sampling_method)
        model = build_model(model_path, max_seq_length)
        train_func(model, model_path, train_dataloader, evaluator, num_hard_negative_queries)
        cls = Candidate_Ranker(model_path=model_path)
        Evaluation.evaluation(cls)