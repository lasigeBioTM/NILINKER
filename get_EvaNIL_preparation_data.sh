#!/usr/bin/env bash

###############################################################################
#                           DOWNLOAD KB FILES                                 #
###############################################################################
mkdir kb_files
cd kb_files

# MEDIC 
wget ctdbase.org/reports/CTD_diseases.obo.gz 
gzip -d CTD_diseases.obo.gz 

# CTD-Chemicals 
wget ctdbase.org/reports/CTD_diseases.tsv.gz 
gzip -d CTD_diseases.tsv.gz 

# CTD-Anatomy 
wget ctdbase.org/reports/CTD_anatomy.tsv.gz 
gzip -d CTD_anatomy.tsv.gz

# ChEBI 
wget ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_lite.obo.gz 
gzip -d chebi_lite.obo.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo.gz 
gzip -d chebi.obo.gz

# GO_BP 
# https://github.com/geneontology/go-site/tree/master/releases
wget http://purl.obolibrary.org/obo/go/go-basic.obo

# HPO 
wget https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo

#Download Disease-gene associations
#wget purl.obolibrary.org/obo/hp/hpoa/phenotype_to_genes.txt

###############################################################################
#                           DOWNLOAD CORPORA                                  #
###############################################################################
cd ../
mkdir corpora
cd corpora/

# CRAFT
wget https://github.com/UCDenver-ccp/CRAFT/archive/v4.0.1.zip 
tar -xvf CRAFT-4.0.1.tar.gz 

# MedMentions
mkdir MedMentions 
cd MedMentions
wget https://github.com/chanzuckerberg/MedMentions/blob/master/full/data/corpus_pubtator.txt.gz 
gzip -d corpus_pubtator.txt.gz 
cd ..

# PubMed DS
#https://drive.google.com/file/d/16mEFpCHhFGuQ7zYRAp2PP3XbAFq9MwoM/view
wget https://drive.google.com/u/0/uc?export=download&confirm=dClU&id=16mEFpCHhFGuQ7zYRAp2PP3XbAFq9MwoM
