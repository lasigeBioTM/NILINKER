import json
#import logging
import matplotlib.pyplot as plt
import numpy as np
import os
#import spacy
import sys
import tensorflow as tf
from fuzzywuzzy import fuzz, process
from src.utils.kbs import KnowledgeBase
sys.path.append("./")


class WordConcept:
    """Class representing a word-concept (WC) dictionary object associated with 
    a given KB. It includes mappings between words and WC int ids, words and 
    candidates, etc.
    """
    
    def __init__(self, partition):
    
        self.partition = partition
        self.filepath = "data/word_concept/wc_" + partition + ".json"
        self.word2candidates = None
        self.word2id = None
        self.id2word = None
        self.candidate_num = None
        self.candidate2id = None
        self.id2candidate = None
        self.root_concept = None
        
    def build(self):
        """Fills a WordConcept object with info about words belonging to 
        the considered Word-Concept (WC) dict, and candidates and respectives 
        int ids. (*note: Candidate int ids are distinct from WC words ids). 
        """

        # Load node_id_to_int
        with open(
                "./data/embeddings/{}/node_id_to_int_{}.json"\
                .format(self.partition, self.partition)) as in_file:
            node_id_to_int = json.load(in_file)

        # Create candidate2id and id2candidate
        id2candidate = dict()
        candidate2id = dict()
        
        for concept in node_id_to_int.keys():
            candidate_int = node_id_to_int[concept]
            candidate2id[concept] = candidate_int 
            id2candidate[candidate_int] = concept

        # Add int for the root concept to wc
        root_dict = {"go_bp": ("GO:0008150", "biological_process"), 
                    "chebi": ("CHEBI:00", "root"), 
                    "hp": ("HP:0000001", "All"), 
                    "medic": ("MESH:C", "Diseases"), 
                    "ctd_anat": ("MESH:A", "Anatomy"), 
                    "ctd_chem": ("MESH:D", "Chemicals")}
        root_concept_kb_id= root_dict[self.partition][0]
        root_concept_int = candidate2id[root_concept_kb_id]
        self.root_concept_int = root_concept_int

        word2candidates = dict()
        
        # Load word-concept file into dict
        with open(self.filepath, 'r', encoding='utf-8') as wc: 
            word2candidates = json.load(wc)
        
        # Create word2candidates with candidates int
        word2candidates_up = dict()
        vocab_size = len(node_id_to_int.keys())
        
        wc_word2id = dict()
        id2word = dict()
        word_count = int()

        for word in word2candidates.keys():
            word_count += 1
            wc_word2id[word] = word_count
            id2word[word_count] = word

            candidates = word2candidates[word]
            candidates_int = list()
            
            for candidate in candidates:
                # Convert concept ID in int
                candidate_int = node_id_to_int[candidate]
                candidate_int_up = int()

                if candidate_int == vocab_size:
                    candidate_int_up = root_concept_int

                else:
                    candidate_int_up = candidate_int
                
                candidates_int.append(candidate_int_up)

            word2candidates_up[word] = candidates_int
            
        self.word2candidates = word2candidates_up
        self.word2id = wc_word2id 
        self.id2word = id2word
        self.candidate2id = candidate2id
        self.id2candidate = id2candidate
        self.candidate_num = len(candidate2id.keys())
        root_concept_kb_id= root_dict[self.partition][0]
        root_concept_int = candidate2id[root_concept_kb_id]
        self.root_concept_int = root_concept_int
        
    
def load_word_embeds(embeds_dir):
    """Loads word embeddings from file into a Numpy array and generate dicts 
    with information about each word and respective int ID.

    :param embeds_filepath: path for the file containing word embeddings
    :type embeds_filepath: str
    
    :return: embeds, word2id, id2word
    :rtype: Numpy array, dict, dict
    """

    embeds = list()
    vocabulary = list()

    with open(embeds_dir + 'word_embeddings.txt', 'r', encoding='utf-8') \
            as embed_file:
        
        for line in embed_file.readlines():
            word = line.split()[0]
            embed = [float(item) for item in line.split()[1:]]
            embeds.append(embed)
            vocabulary.append(word)
        
        embed_file.close()
    
    assert (len(embeds) == len(vocabulary))

    embeds_word2id = dict()
    
    with open(embeds_dir + 'word2id.json', 'r') as word2id_file:
        embeds_word2id = json.load(word2id_file)
        word2id_file.close()
 
    embeds = np.array(embeds)#s.astype('float32')
    embeds = embeds / np.sqrt(np.sum(embeds * embeds, axis=1, keepdims=True))
   
    return embeds, embeds_word2id


def load_candidate_embeds(embeds_filepath, candidate2id):
    """Loads candidate embeddings from file into a Numpy array.

    :param embeds_filepath: path for the file containing candidate 
        (i.e. KB concepts) embeddings
    :type embeds_filepath: str
    :param candidate2id: info about candidates and respective internal ID
    :type candidate2id: dict
    
    :return: embeds
    :rtype: Numpy array
    """

    embed_dict = dict()
    
    with open(embeds_filepath, 'r', encoding='utf-8') as embed_file:
        
        for line in embed_file.readlines():
            word = line.split()[0]
            embedding = [float(item) for item in line.split()[1:]]
            embed_dict[word] = embedding

    embeds = list()
    
    for candidate in candidate2id.keys(): 
        candidate_id = candidate2id[candidate]
        embeds.append(embed_dict[str(candidate_id)])
    
    embeds = np.array(embeds)
    embeds = embeds / np.sqrt(np.sum(embeds * embeds, axis=1, keepdims=True))
    
    return embeds

    
def retrieve_annotations_from_evanil(partition):
    """Loads annotations of chosen partition of EvaNIL dataset in 
    './data/evanil/<partition>' dir into dict.
    
    :param partition: has value 'medic', 'ctd_anat', 'ctd_chem', 
        'chebi', 'go_bp' or 'hp'
    :type partition: str
    
    :return annotations: has format 
        {doc_id: {annotation_str: [kb_id, direct_ancestor_id]}
    :rtype: dict
    """
    
    annotations_filepath = './data/evanil/' +  partition + '.json'
    annotations = dict()

    with open(annotations_filepath, 'r') as in_file:
        annotations = json.load(in_file)
        in_file.close()
    
    return annotations


def retrieve_annotations_into_arrays(partition):
    """Loads annotation from data/annotations/<partition> dir and convert the
    lists into numpy array to further input to NILINKER.
    
    :param partition: has value 'medic', 'ctd_anat', 'ctd_chem', 
        'chebi', 'go_bp' or 'hp'
    :type partition: str
    
    :return: x, y corresponding to input features and expected output.
    :rtype: tuple with 2 numpy arrays
    """
    
    x = list()
    y = list()
    
    annots_dir = 'data/annotations/'
    
    with open(annots_dir + partition + '.json', 'r') as annots_file:
        annotations = json.load(annots_file)
        annots_file.close()

    for annot in annotations:
        x_annot = (annot[0], annot[1], annot[2], annot[3])
        y_annot = (annot[2])
        x.append(x_annot)
        y.append(y_annot)

    x = np.array(x)
    y = np.array(y)
    
    return x, y


def get_candidates_4_word(
        wc_2idword, word2candidates, word_str='', wc_word_id=-1):
    """Retrieves KB candidate concepts for given word (either WC word id or 
    string) or for the most similar word in the WC if the word is not present 
    in the WC.

    :param wc_2idword: mappings between WC word ids and WC words
    :type wc_2idword: dict
    :param word2candidates: mappings between WC words and KB 
        candidate concepts
    :type word2candidates: dict
    :param word_str: string of the target word
    :type word_str = str
    :param wc_word_id: the WC word id for the target word 
    :type wc_word_id: int or numpy.int64

    :raises ValueError: if given input is invalid, input must be either a 
        string or a valid WC word id
    
    :return: candidates_ids
    :rtype: Numpy array

    >>> wc_word_id = 444
    >>> wc_2idword = {444: 'gene'}
    >>> word2candidates = {'gene': [1, 2, 3]}
    >>> get_candidates_4_wordid(wc_2idword,word2candidates,word_id=wc_word_id)
    [1, 2, 3]
    >>> word = 'gene'
    >>> get_candidates_4_wordid(wc_2idword,word2candidates,word_str=word)
    [1, 2, 3]
    >>> word_id = 3.5
    >>> get_candidates_4_wordid(wc_2idword,word2candidates,word_id=word_id)
    ValuError: 'Input not valid! Must be either a word string or a WC word id'
    """

    word = str()
    
    if wc_word_id != -1 and \
            (type(wc_word_id) == np.int64 or type(wc_word_id) == int):
        
        word = wc_2idword[wc_word_id]
        
    else:

        if word_str != '':
            word = word_str
        
        else:
            raise ValueError(
               'Input not valid! Must be either a word string or a WC word id')
    
    candidates = list()

    if word in word2candidates.keys():
        candidates = word2candidates[word]

    else:
        top_match = process.extract(
                        word, word2candidates.keys(), 
                        scorer=fuzz.token_sort_ratio, 
                        limit=1)
        most_similar_word = top_match[0][0]
        candidates = word2candidates[most_similar_word]

    candidates_ids = np.array(candidates)
    
    return candidates_ids

   
def get_wc_embeds(partition):
    """Creates and populates WordConcept object and get Numpy arrays with 
    candidate and word embeddings.

    :param partition: has value 'medic', 'ctd_chem', 'hp', 'chebi', 'go_bp', 
        'ctd_anat'
    :type partition: str
    
    :return: word_embeds, candidate_embeds, wc
    :rtype: Numpy array, Numpy array, WordConcept object
    """

    embeds_dir = "./data/embeddings/{}/".format(partition)
    word_embeds_filepath = embeds_dir 
    candidates_embeds_filepath = embeds_dir + partition + ".emb"

    wc = WordConcept(partition)
    wc.build()

    word_embeds, embeds_word2id = load_word_embeds(word_embeds_filepath)
    
    candidate_embeds = load_candidate_embeds(
                            candidates_embeds_filepath, 
                            wc.candidate2id)

    return word_embeds, candidate_embeds, wc,  embeds_word2id


def get_tokens_4_entity(entity_str):
    
    tokens = list()
    
    if type(entity_str) == str: 
        tokens = entity_str.split(" ")
    
    else:
        raise TypeError('Entity type is not <str>')

    if len(tokens) == 1:
        # If the entity has only 1 word, we assume that 
        # it has two repeated words
        tokens = [tokens[0], tokens[0]]

    return tokens


def get_words_ids_4_entity(
        entity_str, wc_word2id={}, embeds_word2id={}, mode=''):
    """Tokenizes given entity string and, according with the selected mode,
    returns the ids of the words that are part of the entity. If mode 'wc' it 
    returns the WC word ids for left and right words, if mode 'embeds' it 
    returns the embeddings word ids for left and right words. If a given word 
    is not present in the given dictionary, it finds the most similar word in 
    the dict according with the Levenshtein distance.
    
    :param entity_str: the target entity string
    :type entity_str: str
    :param wc_word2id: mappings between each word in the WC and the
        respective int ids
    :type wc_word2id: dict
    :param embeds_word2id: mappings between each word in the embeddings
        vocabulary and the respective int ids
    :type embeds_word2id: dict

    :raise ValueError: if the given mode is invalid, mode must be either 'wc'
        or 'embeds'

    :return: word_l_id and word_r_id including the word ids of the left and
        right words of the given entity.
    :rtype: tuple with 2 ints

    >>> entity = 'arrythmic palpitation'
    >>> wc_word2id = {'palpitation':1663,'arrythmic':1629,'hematoma':1353}
    >>> get_words_ids_4_entity(entity, wc_word2id=wc_word2id, mode='wc')
    1629, 1663
    
    >>> entity = 'arrythmic palpitations'
    >>> embeds_word2id = {'palpitation':1,'arrythmic':2,'hematoma':3}
    >>> mode = 'embeds'
    >>> get_words_ids_4_entity(entity,embeds_word2id=embeds_word2id,mode=mode)
    2, 1
    """
    
    ids = dict()

    if mode == 'wc':
        ids = wc_word2id
    
    elif mode == 'embeds':
        ids = embeds_word2id

    else:
        raise ValueError('Invalid mode! Choose either "wc" (to get the \
                          word ids from Word-Concept) or "embeds" mode \
                         (to get words ids from the embeddings vocabulary)')

    tokens = get_tokens_4_entity(entity_str)

    word_l_id = int()
    word_r_id = int()
   
    token_count = 1
    
    for token in tokens[:2]:
        # Only consider the first two words of the entity
        
        if token in ids.keys():

            if token_count == 1:
                word_l_id = ids[token]

            elif token_count == 2:
                word_r_id = ids[token]

        else:
            top_match = process.extract(
                token, ids.keys(), scorer=fuzz.token_sort_ratio, 
                limit=1)
           
            most_similar_word = ids[top_match[0][0]]
            
            if token_count == 1:
                word_l_id = most_similar_word

            elif token_count == 2:
                word_r_id = most_similar_word
        
        token_count += 1
    
    return word_l_id, word_r_id


def get_kb_data(partition):
    """Loads KB data (concept names, synonyms, IDs, etc) associated with given 
    partition into a KnowledgeBase object.

    :param partition: has value 'medic', 'ctd_chem', 'hp', 'chebi', 
        'go_bp' or 'ctd_anat'
    :type partition: str
    
    :return: kb_data representing the given KB
    :rtype: KnowledgeBase object
    """

    obo_list = ['hp', 'chebi', 'medic', 'go_bp'] 
    tsv_list = ['ctd_chem', 'ctd_anat']
    
    kb_data = KnowledgeBase(partition)
        
    if partition in obo_list:
        kb_data.load_obo(kb_data.kb)  
        
    elif partition in tsv_list:
        kb_data.load_tsv(kb_data.kb)
    
    else:
        raise ValueError('Invalid KB/partition! Input partition must be \
                         medic, ctd_chem, hp, chebi, go_bp or ctd_anat')

    return kb_data