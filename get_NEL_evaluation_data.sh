#!/usr/bin/env bash

###############################################################################
#                   Get datasets to evaluate the NEL models:
###############################################################################

# CHR corpus
wget www.nactem.ac.uk/CHR/CHR_corpus.tar.gz
tar xvzf CHR_corpus.tar.gz

# GSC+ corpus
mkdir GSC+
cd GSC+
wget https://github.com/lasigeBioTM/IHP/raw/master/GSC%2B.rar
unrar x GSC+.rar
cd ../

# Download PHAEDRA corpus
wget www.nactem.ac.uk/PHAEDRA/PHAEDRA_corpus.tar.gz
tar -xvf PHAEDRA_corpus.tar.gz