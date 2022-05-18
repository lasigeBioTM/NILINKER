import json
import gensim
import logging
import os
import spacy
import sys
from src.utils.kbs import KnowledgeBase
from src.utils.utils import retrieve_annotations_from_evanil, get_tokens_4_entity
from tqdm import tqdm
sys.path.append("./")


def build_node2vec_input(partition):
    """Prepares the input .json files to node2vec to further build candidate 
    embeddings. Candidate embedings are node2vec embeddings for the KB concepts
    associated with given partition. It outputs 3 json files:
      
        - int_to_node_id.json in dir 'data/embeddings/<partition>/'
        - node_id_to_int.json in dir 'data/embeddings/<partition>'
        - <partition>.edgelist in dir 'node2vec/graph/'

    :param partition: has value 'medic', 'ctd_anat', 'ctd_chem', 
        'chebi', 'go_bp' or 'hp'
    :type partition: str            
    """        

    logging.info('-----> Preparing input for node2vec...')
    
    kb_data = KnowledgeBase(partition)
    obo_list = ['hp', 'chebi', 'medic', 'go_bp'] 
    tsv_list = ['ctd_chem', 'ctd_anat']
    
    if partition in obo_list:
        kb_data.load_obo(kb_data.kb)

    elif partition in tsv_list:
        kb_data.load_tsv(kb_data.kb)

    output = str()
    node_count = int()
    int_to_node = dict()
    node_to_int = dict()

    for node in kb_data.name_to_id.keys():
        node_id = kb_data.name_to_id[node]
        # Assign an internal ID to each KB concept
        node_count += 1 
        int_to_node[node_count] = node_id
        node_to_int[node_id] = node_count
        
    # Node2vec does not handle KB ids, so it is necessary to convert them to ints
    embeds_dir = "./data/embeddings/{}/".format(partition)

    if not os.path.exists(embeds_dir):
        os.makedirs(embeds_dir)

    filepath_1 = embeds_dir + "int_to_node_id_{}.json".format(partition)

    out = json.dumps(int_to_node)

    with open(filepath_1, 'w') as out_file:
        out_file.write(out)
        out_file.close()
    
    filepath_2 = embeds_dir + "node_id_to_int_{}.json".format(partition)

    out_2 = json.dumps(node_to_int)

    with open(filepath_2, 'w') as out_file2:
        out_file2.write(out_2)
        out_file2.close()
    
    for edge in kb_data.edges:
        
        if edge[0] in node_to_int.keys() \
                and edge[1] in node_to_int.keys():
            edge_1 = node_to_int[edge[0]]
            edge_2 = node_to_int[edge[1]]
            output += str(edge_1) + ' ' + str(edge_2) + "\n"

    with open('./node2vec/graph/' + partition + '.edgelist', 'w') as out_file3:
        out_file3.write(output)
        out_file3.close()


def generate_bio2vec_word_embeds(partition):
    """Creates word embeddings file for given partition. Word embeddings are 
    bio2vec embeddings associated with the words present in the selected 
    partition of EvaNIL and with the words present in the Word-Concept dict 
    for the same partition. It outputs the file word_embeddings.txt in dir 
    'data/embeddings/<partition>'.

    :param partition: has value 'medic', 'ctd_anat', 'ctd_chem', 'chebi', 
        'go_bp' or 'hp'
    :type partition: str
    """        
    
    logging.info('-----> Generating word_embeddings.txt for {}...'.\
        format(partition))

    logging.info('-----> Loading gensim bio2vec model...')

    bio2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        './data/embeddings/bio2vec/bio_embedding_intrinsic', 
        binary=True, limit=int(4E7))
        
    annotations = retrieve_annotations_from_evanil(partition, 'train')
    dev_annotations = retrieve_annotations_from_evanil(partition, 'dev')
    test_annotations = retrieve_annotations_from_evanil(partition, 'test')
    annotations.update(dev_annotations)
    annotations.update(test_annotations)

    logging.info('-----> Building embeddings for words in annotations...')

    output = str()
    added_tokens = list()
    token_count = int()
    embeds_word2id = dict()
    embeds_id2word = dict()
    nlp = spacy.load('en_core_sci_md')
    pbar = tqdm(total=len(annotations.keys()))

    for doc in annotations.keys():
        doc_annotations = annotations[doc]
        
        for annot in doc_annotations.keys():
            # To prevent repeated entities in the same document
            tokens = get_tokens_4_entity(annot)

            for token in tokens:

                if token not in added_tokens:
                    added_tokens.append(token)
                    
                    # Get the lemma of the word
                    token_ = nlp(token)
                    token_lemma = token_[0].lemma_

                    if token_lemma in bio2vec_model:
                        token_count += 1
                        embeds_word2id[token_lemma] = token_count
                        embeds_id2word[token_count] = token_lemma

                        output += token_lemma

                        for dim in bio2vec_model[token_lemma]:
                            output += ' ' + str(dim)
                       
                        output += '\n' 

                    else:
                        # Out-of-vocabulary word
                        continue 
        pbar.update(1)
    
    pbar.close()

    logging.info('-----> Outputting files...')

    embeds_dir = "./data/embeddings/{}/".format(partition)

    if not os.path.exists(embeds_dir):
        os.makedirs(embeds_dir)

    embeds_filepath = embeds_dir + 'word_embeddings.txt'

    embed_file = open(embeds_filepath, 'w') 
    embed_file.write(output)
    embed_file.close()

    word2id_json = json.dumps(embeds_word2id, indent=2)
    word2id_file = open(embeds_dir + 'word2id.json', 'w')
    word2id_file.write(word2id_json)
    word2id_file.close()

    id2word_json = json.dumps(embeds_id2word, indent=2)
    id2word_file = open(embeds_dir + 'id2word.json', 'w')
    id2word_file.write(id2word_json)
    id2word_file.close()

    logging.info('-----> Done!')


if __name__ == "__main__":
    partition = sys.argv[1]
    
    log_dir = './logs/{}/'.format(partition)
    log_filename = log_dir + 'embeddings.log'
    logging.basicConfig(
        filename=log_filename, level=logging.INFO, 
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
        filemode='w')

    generate_bio2vec_word_embeds(partition) 
    
    build_node2vec_input(partition) 