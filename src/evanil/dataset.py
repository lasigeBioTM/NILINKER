import argparse
import json
import logging
import os
import random
import sys
from annotations import parse_annotations
from src.utils.kbs import KnowledgeBase

sys.path.append("./")


def refine_test_set(train_json, dev_json, test_json, test_pubtator):
    """Refines test set by removing entity mentions that are present in the 
    train and dev sets. It corresponds to the stratified setting described in 
    the article 'Fair Evaluation in Concept Normalization: a Large-scale 
    Comparative Analysis for BERT-based Models'"""

    # Import train and dev mentions into list 
   
    train_dev_mentions = [
        mention for doc_id in train_json for mention in train_json[doc_id].keys()]
        
    for doc_id in dev_json:

        for mention in dev_json[doc_id]:
            train_dev_mentions.append(mention)

    #-------------------------------------------------------------------------
    # Filter out repeated mentions from test json
    #-------------------------------------------------------------------------
    test_refined_json = {}

    for doc_id in test_json.keys():

        for mention in test_json[doc_id]:

            if mention not in train_dev_mentions:
                # Include this mention in the refined test set
        
                if doc_id in test_refined_json.keys():
                    test_refined_json[doc_id][mention] = test_json[doc_id][mention]
                    
                else:
                    test_refined_json[doc_id] = {mention: test_json[doc_id][mention]}

    #-------------------------------------------------------------------------
    # Filter out repeated mentions from test pubtator
    #-------------------------------------------------------------------------
    test_pubtator_data = test_pubtator.split('\n')
    test_refined_pubtator = ''
    doc_count = []
    is_mention = True
    added_doc = False
    ignore_doc = False
    tmp_doc = ''
    
    for line in test_pubtator_data:
        doc_id = line.split('\t')[0]
         
        if '|' in doc_id:
            # Title and abstract
            is_mention = False
            ignore_doc = False
            doc_id = doc_id.split('|')[0]
                
            if '|t|' in line:
                # This line is the tile
                tmp_doc = ''
                tmp_doc += line
             
            tmp_doc += line + '\n'
            added_doc = False
           
        else:
            is_mention = True

        if is_mention and line != '': 
            
            if line[0].isalpha():
                ignore_doc = True
            
            else:

                try:
                    mention = line.split('\t')[3] 
                
                    if mention not in train_dev_mentions and not ignore_doc:
                        
                        if not added_doc:
                            # Add text + abstract
                            test_refined_pubtator += '\n' + tmp_doc 
                            added_doc = True
                            tmp_doc = ''
                    
                        test_refined_pubtator += line + '\n'

                        if doc_id not in doc_count:
                            doc_count.append(doc_id)
                    
                    mention = ''
                
                except:
                    continue
    
    return test_refined_json, test_refined_pubtator


def resize_dataset(
        partition, train_json, train_pubtator, dev_json, dev_pubtator, test_json, 
        test_pubtator):
    """Structures annotations parsed from PBDMS dataset into a ratio of 
    70% train, 15% dev and 15% test"""

    if partition == 'medic':
        cuttoff = 61999

    elif partition == 'ctd_anat':
        cuttoff = 61999

    elif partition == 'ctd_chem':
        cuttoff = 3999
        added_2_train = 0

    #-------------------------------------------------------------------------
    # Resize json dicts
    #-------------------------------------------------------------------------
    train_json_up = train_json
    dev_json_up = {}
    test_json_up = {}
    ignore_docs = []

    for i, doc_id in enumerate(dev_json):
    
        if i <= cuttoff and doc_id not in ignore_docs:
            dev_json_up[doc_id] = dev_json[doc_id]
        
        else:

            if partition == 'medic' or partition == 'ctd_anat' or \
                    (partition == 'ctd_chem' and added_2_train<= 2000) \
                    and doc_id not in ignore_docs:

                train_json_up[doc_id] = dev_json[doc_id]

                if partition == 'ctd_chem':
                    added_2_train+= 1

    added_2_train = 0

    for i, doc_id in enumerate(test_json):
        
        if i <= cuttoff and doc_id not in ignore_docs:
            test_json_up[doc_id] = test_json[doc_id]
        
        else:

            if partition == 'medic' or partition == 'ctd_anat' or \
                    (partition == 'ctd_chem' and added_2_train<= 2000) \
                    and doc_id not in ignore_docs:

                train_json_up[doc_id] = test_json[doc_id]

                if partition == 'ctd_chem':
                    added_2_train+= 1

    print('Train:', len(train_json_up))
    print('Dev:', len(dev_json_up))
    print('Test:', len(test_json_up))

    #-------------------------------------------------------------------------
    # Resize pubtator
    #-------------------------------------------------------------------------
    train_pubtator_up = train_pubtator
    
    dev_pubtator_data = dev_pubtator.split('\n')
    dev_pubtator_up = ''
    doc_count = []
    added_2_train = 0

    for line in dev_pubtator_data:
        doc_id = line.split('\t')[0]

        if '|' in doc_id:
            doc_id = doc_id.split('|')[0]
        
        if doc_id not in doc_count:
            doc_count.append(doc_id)
        
        if len(doc_count) <= cuttoff and doc_id not in ignore_docs:
            dev_pubtator_up += line + '\n'
        
        else:

            if partition == 'medic' or partition == 'ctd_anat' or \
                    (partition == 'ctd_chem' and added_2_train<= 2000) \
                    and doc_id not in ignore_docs:

                train_pubtator_up += line + '\n'

                if partition == 'ctd_chem':
                    added_2_train+= 1

    test_pubtator_data = test_pubtator.split('\n')
    test_pubtator_up  = ''
    doc_count = []
    added_2_train = 0

    for line in test_pubtator_data:
        doc_id = line.split('\t')[0]

        if '|' in doc_id:
            doc_id = doc_id.split('|')[0]
        
        if doc_id not in doc_count:
            doc_count.append(doc_id)

        if len(doc_count) <= cuttoff and doc_id not in ignore_docs:
            test_pubtator_up += line + '\n'
        
        else:

            if partition == 'medic' or partition == 'ctd_anat' or \
                    (partition == 'ctd_chem' and added_2_train<= 2000) \
                    and doc_id not in ignore_docs:

                train_pubtator_up += line + '\n'

                if partition == 'ctd_chem':
                    added_2_train+= 1

    #-------------------------------------------------------------------------
    # Refine test set, i.e. eliminate mentions from test set that are present
    # in train and dev sets
    #-------------------------------------------------------------------------
    test_refined_json, test_refined_pubtator = refine_test_set(
        train_json_up, dev_json_up, test_json_up, test_pubtator_up)

    return train_json_up, train_pubtator_up, dev_json_up, dev_pubtator_up, \
            test_json_up, test_pubtator_up, test_refined_json, test_refined_pubtator


def split_dataset(json_data, pubtator_data):
    """For non-PBDMS documents"""

    # Split document ids to train and test sets
    doc_ids = json_data.keys()
    num_docs = len(doc_ids)
    train_size = int(0.70 * num_docs)
    dev_size = int(0.15 * num_docs)
    test_size = int(0.15 * num_docs)

    diff = train_size + dev_size + test_size - num_docs
    
    if diff > 0:
        train_size -= diff
    
    elif diff < 0:
        train_size += abs(diff)
    
    assert train_size + dev_size + test_size == num_docs

    random.seed(10)
    train_ids = random.sample(doc_ids, train_size)
    add_test = True
    dev_ids = []
    test_ids = []

    for doc_id in doc_ids:

        if doc_id not in train_ids:

            if add_test:
                add_test = False
                test_ids.append(doc_id)

            else:
                dev_ids.append(doc_id)
                add_test = True

    # Populate train and test json dicts
    train_json = {}
    dev_json = {}
    test_json = {}
    
    for doc_id in json_data:

        if doc_id in train_ids:
            train_json[doc_id] = json_data[doc_id]

        elif doc_id in dev_ids:
            dev_json[doc_id] = json_data[doc_id]

        elif doc_id in test_ids:
            test_json[doc_id] = json_data[doc_id]

    pubtator_lines = pubtator_data.split('\n')
    train_pubtator = ''
    dev_pubtator = ''
    test_pubtator = ''
    
    for line in pubtator_lines:
        doc_id = line.split('\t')[0]
        
        if '|' in doc_id:
            doc_id = doc_id.split('|')[0]
        
        if doc_id in train_ids:

            if '|t|' in line:
                train_pubtator += '\n'

            train_pubtator += line + '\n'

        if doc_id in dev_ids:

            if '|t|' in line:
                dev_pubtator += '\n'

            dev_pubtator += line + '\n'
        
        elif doc_id in test_ids:

            if '|t|' in line:
                test_pubtator += '\n'

            test_pubtator += line + '\n'

    return train_json, train_pubtator, dev_json,  dev_pubtator, test_json, test_pubtator


def build_partition(partition):
    """Generates the <partition>.json file in the dir 'data/evanil/ containing
    the partition annotations.

    :param annotations: has format 
        {file_id: {annotation_str: [KB_id, direct_ancestor]}}
    :type annotations: dict
    :param partition: has value 'medic', 'ctd_chem', 'chebi', 'hp', 'ctd_anat'
    :type partition: str
    """

    logging.info('Parsing annotations...')

    obo_list = ['hp', 'chebi', 'medic', 'go_bp'] 
    tsv_list = ['ctd_chem', 'ctd_anat']
    kb_data = KnowledgeBase(partition)

    if partition in obo_list:
        kb_data.load_obo(kb_data.kb)  

    elif partition in tsv_list:    
        kb_data.load_tsv(kb_data.kb)      
  
    has_pbmds_files = ['medic', 'ctd_anat', 'ctd_chem']

    if partition in has_pbmds_files:   
        # PBDMS dataset is too large, so each split
        # is processed sequentially.
        
        #---------------------------------------------------------------------
        # To generate only the splits used in the experiments
        splits_dict = {
            'medic': {'train': ['0', '7', '14', '21'], 'dev': '16', 'test': '27'},
            'ctd_anat': {'train': ['0', '7', '14', '21'], 'dev': '16', 'test': '27'},
            'ctd_chem': {'train': '0', 'dev': '16', 'test': '27'} 
            }

        splits = splits_dict[partition]
        
        #---------------------------------------------------------------------       
        train_json_tmp = {}
        train_pubtator_tmp = ''

        for split in splits['train']:
            out_dict, pubtator_output = parse_annotations(
                                    kb_data, partition, split=split)
            train_json_tmp.update(out_dict)
            train_pubtator_tmp += pubtator_output
        
        dev_json_tmp, dev_pubtator_tmp = parse_annotations(
            kb_data, partition, split=splits['dev'])
        test_json_tmp, test_pubtator_tmp = parse_annotations(
            kb_data, partition, split=splits['test'])

        # Adjust dimension of train, dev and test sets (70%, 15%, 15%)
        train_json, train_pubtator, \
            dev_json, dev_pubtator, \
            test_json, test_pubtator,\
            test_refined_json, test_refined_pubtator = resize_dataset(
                    partition, train_json_tmp, train_pubtator_tmp, 
                    dev_json_tmp, dev_pubtator_tmp, test_json_tmp, 
                    test_pubtator_tmp)

    else:
        out_dict, pubtator_output = parse_annotations(kb_data, partition)

        # Adjust dimension of train, dev and test sets (70%, 15%, 15%)
        train_json, train_pubtator, \
            dev_json, dev_pubtator, \
            test_json, test_pubtator = split_dataset(out_dict, pubtator_output)

        # refine test set
        test_refined_json, test_refined_pubtator = refine_test_set(
            train_json, dev_json, test_json, test_pubtator)

    #--------------------------------------------------------------------------
    #                       OUTPUT JSON files
    #--------------------------------------------------------------------------
    out_dir = "./data/evanil/"     

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_dir += partition + '/'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    train_json_out = json.dumps(train_json, indent=4, ensure_ascii=False)
    train_json_filename = out_dir + "train.json"

    with open(train_json_filename, "w", encoding="utf-8") as train_json_file:
        train_json_file.write(train_json_out)
        train_json_file.close()
        
    dev_json_out = json.dumps(dev_json, indent=4, ensure_ascii=False)
    dev_json_filename = out_dir + "dev.json"

    with open(dev_json_filename, "w", encoding="utf-8") as dev_json_file:
        dev_json_file.write(dev_json_out)
        dev_json_file.close()
    
    test_json_out = json.dumps(test_json, indent=4, ensure_ascii=False)
    test_json_filename = out_dir + "test.json"

    with open(test_json_filename, "w", encoding="utf-8") as test_json_file:
        test_json_file.write(test_json_out)
        test_json_file.close()
    
    test_refined_json
    test_refined_json_out = json.dumps(test_refined_json, indent=4, ensure_ascii=False)
    test_refined_json_filename = out_dir + "test_refined.json"

    with open(test_refined_json_filename, "w", encoding="utf-8") as test_refined_json_file:
        test_refined_json_file.write(test_refined_json_out)
        test_refined_json_file.close()
    
    #--------------------------------------------------------------------------
    #                       OUTPUT Pubtator files
    #--------------------------------------------------------------------------
    train_pubtator_filename = out_dir + "train.txt"
    
    with open(train_pubtator_filename, "w") as train_pubtator_file:
        train_pubtator_file.write(train_pubtator)
        train_pubtator_file.close()
    
    dev_pubtator_filename = out_dir + "dev.txt"
    
    with open(dev_pubtator_filename, "w") as dev_pubtator_file:
        dev_pubtator_file.write(dev_pubtator)
        dev_pubtator_file.close()

    test_pubtator_filename = out_dir + "test.txt"
    
    with open(test_pubtator_filename, "w") as test_pubtator_file:
        test_pubtator_file.write(test_pubtator)
        test_pubtator_file.close()

    test_refined_pubtator_filename = out_dir + "test_refined.txt"
    
    with open(test_refined_pubtator_filename, "w") as test_refined_pubtator_file:
        test_refined_pubtator_file.write(test_refined_pubtator)
        test_refined_pubtator_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                prog="dataset", 
                description="Builds the selected partition of the EvaNIL \
                            dataset")
    parser.add_argument(
        "-partition", 
        type=str, 
        choices=['hp', 'chebi', 'medic', 'ctd_chem', 'ctd_anat', 'go_bp'],
        help="The seleccted partition to build.")               
    args = parser.parse_args()
    
    build_partition(args.partition)