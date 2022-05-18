#!/usr/bin/env bash


###############################################################################
#                   DOWNLOAD DATA TO USE NILINKER                             #
###############################################################################
mkdir data
cd data/

# Word-concept dictionaries
wget https://zenodo.org/record/5927300/files/word_concept.tar.gz?download=1
tar -xvf word_concept.tar.gz?download=1
rm 'word_concept.tar.gz?download=1'

# Trained models files
wget https://zenodo.org/record/5927300/files/nilinker_files.tar.gz?download=1
tar -xvf nilinker_files.tar.gz?download=1
rm 'nilinker_files.tar.gz?download=1'

# Embeddings
wget https://zenodo.org/record/5927300/files/embeddings.tar.gz?download=1
tar -xvf embeddings.tar.gz?download=1
rm 'embeddings.tar.gz?download=1'

#wget https://zenodo.org/record/5927300/files/REEL.tar.gz?download=1
#tar -xvf REEL.tar.gz?download=1
#rm 'REEL.tar.gz?download=1'