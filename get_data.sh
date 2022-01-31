#!/usr/bin/env bash


###############################################################################
#                   DOWNLOAD DATA TO USE NILINKER                             #
###############################################################################
mkdir data
cd data/

wget https://zenodo.org/record/5927300/files/word_concept.tar.gz?download=1
tar -xvf word_concept.tar.gz

wget https://zenodo.org/record/5927300/files/word_concept.tar.gz?download=1
tar -xvf nilinker_files.tar.gz

wget https://zenodo.org/record/5927300/files/embeddings.tar.gz?download=1
tar -xvf embeddings.tar.gz

wget https://zenodo.org/record/5927300/files/word_concept.tar.gz?download=1
tar -xvf annotations.tar.gz

wget https://zenodo.org/record/5927300/files/embeddings.tar.gz?download=1
tar -xvf REEL.tar.gz

wget https://zenodo.org/record/5927300/files/evanil_to_train.tar.gz?download=1
tar -xvf
mv evanil_to_train evanil


###############################################################################
#                           DOWNLOAD KB FILES                                 #
###############################################################################
mkdir kb_files
cd kb_files

# MEDIC (2021-09-29 10:33)
wget ctdbase.org/reports/CTD_diseases.obo.gz 
gzip -d CTD_diseases.obo.gz 

# CTD-Chemicals (2021-09-29 10:33)
wget ctdbase.org/reports/CTD_diseases.tsv.gz 
gzip -d CTD_diseases.tsv.gz 

# CTD-Anatomy (2021-09-29 12:57)
wget ctdbase.org/reports/CTD_anatomy.tsv.gz 
gzip -d CTD_anatomy.tsv.gz

# ChEBI (13-Oct-2021 14:23) UPDATE
wget ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_lite.obo.gz 
gzip -d chebi_lite.obo.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo.gz 
gzip -d chebi.obo.gz

# GO_BP (2021-09-01) Release notes: https://github.com/geneontology/go-site/tree/master/releases
wget http://purl.obolibrary.org/obo/go/go-basic.obo

# HPO (2021-10-10 release)
wget https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo
#Download Disease-gene associations
wget purl.obolibrary.org/obo/hp/hpoa/phenotype_to_genes.txt


###############################################################################
#                           DOWNLOAD CORPORA                                  #
###############################################################################
cd ../
mkdir corpora
cd corpora/

# CRAFT
wget https://github.com/UCDenver-ccp/CRAFT/archive/v4.0.1.zip 
tar -xvf CRAFT-4.0.1.tar.gz 

# NCBI Disease corpus
mkdir NCBI_disease_corpus
cd NCBI_disease_corpus
wget https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip
unzip NCBIdevelopset_corpus.zip
wget https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip
unzip NCBItestset_corpus.zip
wget https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip
unzip NCBItrainset_corpus.zip
cd ..

# MedMentions
mkdir MedMentions 
cd MedMentions
wget https://github.com/chanzuckerberg/MedMentions/blob/master/full/data/corpus_pubtator.txt.gz 
gzip -d corpus_pubtator.txt.gz 
cd ..


# BC5CDR
wget https://codeload.github.com/JHnlp/BioCreative-V-CDR-Corpus/zip/master 
unzip master

# CHR corpus
wget www.nactem.ac.uk/CHR/CHR_corpus.tar.gz
tar xvzf CHR_corpus.tar.gz

# GSC+ corpus
wget https://github.com/lasigeBioTM/IHP/raw/master/GSC%2B.rar
unzip GSC+.rar

# PubMed DS
#https://drive.google.com/file/d/16mEFpCHhFGuQ7zYRAp2PP3XbAFq9MwoM/view
wget https://drive.google.com/u/0/uc?export=download&confirm=dClU&id=16mEFpCHhFGuQ7zYRAp2PP3XbAFq9MwoM

# Download PHAEDRA corpus
wget www.nactem.ac.uk/PHAEDRA/PHAEDRA_corpus.tar.gz
tar -xvf PHAEDRA_corpus.tar.gz


# OPTIONAL: only if it is necessary to generate again the embeddings files

###############################################################################
#                        Get node2vec repository                              #
###############################################################################
#cd ..
#git clone https://github.com/aditya-grover/node2vec.git


###############################################################################
#                           DOWNLOAD BIO2VEC EMBEDDINGS                       
#   (Only necessary if generating all file from scratch instead of downloading
#                       the already generated files)
###############################################################################
#cd ..
#cd embeddings
#wget https://figshare.com/ndownloader/articles/6882647/versions/2

