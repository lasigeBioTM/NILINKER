#!/usr/bin/env bash

###############################################################################
#                           PREPARE NILINKER                                  #
#      (only needed in the first time using the model or given dataset)       #
#      Args:                                                                  #
#             $1 - partition                                                  #
#                                                                             #
###############################################################################


echo '» Generating Word-Concept dictionary...'
python src/utils/word_concept.py $1
echo '» Done! Word-Concept dictionary in dir data/word_concept/'

 
echo '» Generating word embeddings and preparing input to node2vec...' 
python src/utils/embeddings.py $1   
echo '» Done!'

echo '» Generating candidate embeddings with node2vec...'
#git clone https://github.com/aditya-grover/node2vec.git
python node2vec/src/main.py --input node2vec/graph/$1.edgelist --output ./data/embeddings/$1/$1.emb --dimensions 200 --directed
echo '» Done! Word and candidate embeddings in dir data/embeddings/'$1
    

echo '» Generating annotations to train NILINKER...'
python src/utils/annotations.py $1
echo '» Done! annotations in data/annotations/'$1 dir


mkdir logs/