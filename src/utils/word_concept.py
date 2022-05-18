import json
import logging
import spacy
import sys
import src.utils.kbs as kbs
from tqdm import tqdm

sys.path.append("./")


def build_word_concept_file(partition):
    """Builds a file including all words appearing in concept names from the 
    target KB (the respective lemmas) associated with the respective KB ids 
    where they appear.

    :param partition: :param partition: has value 'medic', 'ctd_chem', 
        'hp', 'chebi', 'go_bp', 'ctd_anat'
    :type partition: str

    :return: wc_<partition>.json file in 'data/word_concept/' dir
    """

    kb_data = kbs.KnowledgeBase(partition)
    obo_list = ['hp', 'chebi', 'medic', 'go_bp'] 
    tsv_list = ['ctd_chem', 'ctd_anat']
    
    if partition in obo_list:
        kb_data.load_obo(kb_data.kb)

    elif partition in tsv_list:
        kb_data.load_tsv(kb_data.kb)

    logging.info('-----> Building the word concept-dict...')

    # Get the words associated with the concepts and 
    # associate them with the respective concept ids  
    name_to_id = kb_data.name_to_id
    
    synonym_to_id = kb_data.synonym_to_id
    all_concepts = {**name_to_id, **synonym_to_id}

    word_concept = dict()
    
    nlp = spacy.load('en_core_sci_md')
    
    pbar = tqdm(total=len(all_concepts.keys()))

    for concept_name in all_concepts.keys():
        KB_id = all_concepts[concept_name] 
        tokens = concept_name.split(" ")

        for token in tokens: 
            
            if token != " ":

                if len(token) >= 3: 
                    # To avoid useless (stop) words

                    tokens_to_lemma = list()

                    if partition == 'chebi':
                        # Further split the tokens since many chemical names
                        # include words separated by hyphens
                        tokens_to_lemma = token.split('-')
                    
                    else:
                        tokens_to_lemma = [token]
                    
                    for token in tokens_to_lemma:
                        
                        if len(token) >= 3:
                            token_ = nlp(token)
                            token_lemma = token_[0].lemma_ 
                                    
                            if token_lemma not in word_concept.keys():
                                word_concept[token_lemma] = [KB_id]
                                
                            elif token_lemma in word_concept.keys():
                                # To prevent repetition of words in 
                                # word_concept keys
                                token_KB_ids = word_concept[token_lemma]

                                if KB_id not in token_KB_ids: 
                                    token_KB_ids.append(KB_id)
                                    word_concept[token_lemma] = token_KB_ids
       
        pbar.update(1)
    
    pbar.close()

    logging.info('-----> Outputting file...')
    print(len(word_concept))
    json_dict = json.dumps(word_concept)
    json_file = open("data/word_concept/wc_" + partition +".json", "w")
    json_file.write(json_dict)
    json_file.close()
    
    logging.info('-----> Done!')
    
    logging.info("{} words were associated with {} IDs".\
        format(len(word_concept.keys()), kb_data.kb))

    
if __name__ == "__main__":
    partition = str(sys.argv[1])

    log_dir = './logs/{}/'.format(partition)
    log_filename = log_dir + 'word_concept.log'
    logging.basicConfig(
        filename=log_filename, level=logging.INFO, 
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
        filemode='w')
    
    build_word_concept_file(partition)