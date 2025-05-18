You need to add the flax_model.msgpack, model.safetensors, pytorch_model.bin, and tf_model.h5 files to the TGCN/2layers/models/bert directory. Also, you need to import stanford-corenlp in the root directory.

Dependencies:
- Torch 1.9.1
- Python 3.7
- CUDA 11.1
- sentence_transformers 2.2.2
- transformers 4.30.2
- StanfordCoreNLP 3.8.0.1

Code Explanation:
- gcn.py: It contains graph convolutional layers and GCN (Graph Convolutional Network). In GCN, there are operations for in-graph propagation as well as operations on semantic and syntactic aspects.
- gcn1.py: It defines the operations for inter-graph propagation.
- find.py: It is the training process, which calls various functions such as loading domain knowledge and dialogue sets, mining hard negative samples, and performing training.
- candidate_ranker.py: It is used for mining hard negative samples and interval sampling.
- Syntactic.py: It is for syntactic analysis.
- Semantic.py: It is for semantic analysis.
- build_graph.py: It builds semantic graphs, syntactic graphs, and sequential graphs.
- BidirectionalHardNegativesRankingLoss.py: It defines the loss function.
- Evaluation.py and evaluation_metrics.py: They are for validation and evaluation.
- test.py, test.1py, test2.py, and test3.py: They contain functions such as testing, data processing, and storage procedures. 
