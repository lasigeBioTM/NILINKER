

# OPTIONAL: only if it is necessary to generate again the embeddings files

###############################################################################
#                        Get node2vec repository                              #
###############################################################################
git clone https://github.com/aditya-grover/node2vec.git


###############################################################################
#                           DOWNLOAD BIO2VEC EMBEDDINGS                       
#   (Only necessary if generating all file from scratch instead of downloading
#                       the already generated files)
###############################################################################
cd data/embeddings
wget https://figshare.com/ndownloader/articles/6882647/versions/2
unzip 6882647.zip
rm 6882647.zip