import json
#import logging
import os
import sys
from tqdm import tqdm
from utils import retrieve_annotations_from_evanil, get_words_ids_4_entity, WordConcept
sys.path.append('/')


def generate_annotations_file(
        partition, subset, wc_word2id, candidate2id, embeds_word2id):
    """Preprocesses the annotations of given partition and creates the input
    file for training and evaluation of NILINKER, which includes the WC word 
    ids and the embeddings words of the left and right words of the
    original entity and the respective candidate id associated with the gold 
    label. The file is outputted as 'data/annotations/<partition>.json'. 
    Each annotation has the following format:
        
                    [wc_word_l_id, wc_word_r_id,
                    embeds_word_l_id, embeds_word_r_id,
                    gold_label_id]

    The first line represents the WC word ids of a given entity.
    The second line represents the emebddings word ids of a given entity.
    The third line represents the KB candidate id associated with the gold
    label for the given entity, which corresponds
    """

    #.info('-----> Retrieving annotations from evanil...')

    annotations = retrieve_annotations_from_evanil(partition, subset)

    #logging.info('-----> Converting each annotation into word ids...')

    all_annots_2_output = list()
    pbar = tqdm(total=len(annotations.keys()))
    
    for doc in annotations.keys():
        doc_annotations = annotations[doc]
        added_annots = list()
        
        for annot in doc_annotations.keys():
            # To prevent repeated entities in the same document
            entity = annot
            out_annot = list()
            
            if  entity not in added_annots:
                added_annots.append(entity)
                
                # Retrieve WC word ids
                wc_word_l_id, wc_word_r_id = get_words_ids_4_entity(
                                                entity, 
                                                wc_word2id=wc_word2id,
                                                mode = 'wc')
                
                out_annot.append(wc_word_l_id)
                out_annot.append(wc_word_r_id)

                # Retrieve embeddings word ids
                embeds_word_l_id, \
                    embeds_word_r_id = get_words_ids_4_entity(
                                            entity, 
                                            embeds_word2id=embeds_word2id,
                                            mode = 'embeds')
                
                out_annot.append(embeds_word_l_id)
                out_annot.append(embeds_word_r_id)
                
                # The KB concept that should disambiguate the entity exists
                # but we will assume that the concept does not exist
                # Since the true KB concept does not exist, current
                # entity must be linked to the direct ancestor of 
                # the true KB concept
                
                gold_label = doc_annotations[annot][1]
                gold_label_id = candidate2id[gold_label]
                #print(gold_label_id)
                if gold_label_id > 1807:
                    print(gold_label_id)
    
                out_annot.append(gold_label_id)
        
                all_annots_2_output.append(out_annot)
        
        pbar.update(1)
    
    pbar.close()
    
    #logging.info('-----> Outputting file...')

    out_dir = 'data/annotations/{}/'.format(partition)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    out_json = json.dumps(all_annots_2_output, indent=3)

    with open(out_dir + subset + '.json', 'w') as out_file:
        out_file.write(out_json)
        out_file.close()
    
    #logging.info('-----> Done!')


if __name__ == '__main__':
    partition = sys.argv[1]

    """
    log_dir = './logs/{}/'.format(partition)
    log_filename = log_dir + 'annotations.log'
    logging.basicConfig(
        filename=log_filename, level=logging.INFO, 
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
        filemode='w')
    """
    wc = WordConcept(partition)
    wc.build()

    embeds_word2id = dict()
    embeds_filepath = 'data/embeddings/{}/'.format(partition)
    
    with open(embeds_filepath + 'word2id.json', 'r') as embeds_word2id_file:
        embeds_word2id = json.load(embeds_word2id_file)
        embeds_word2id_file.close()

    generate_annotations_file(partition, 
                              'train',
                              wc.word2id, 
                              wc.candidate2id,
                              embeds_word2id)

    generate_annotations_file(partition, 
                              'dev',
                              wc.word2id, 
                              wc.candidate2id,
                              embeds_word2id)
    
    #generate_annotations_file(partition, 
    #                          'test',
    #                          wc.word2id, 
    #                          wc.candidate2id,
    #                          embeds_word2id)
    
    #generate_annotations_file(partition, 
    #                          'test_refined',
    #                          wc.word2id, 
    #                          wc.candidate2id,
    #                          embeds_word2id)