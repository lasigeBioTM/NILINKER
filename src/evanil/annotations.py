import csv
import json
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from src.utils import kbs
from tqdm import tqdm
sys.path.append('./')


def add_or_ignore_annot(doc_id, annotation, current_dict):
    """Updates output dict with given annotation if not yet associated with 
    given doc_id.
    
    :param doc_id: the identification of the document
    :type doc_id: str
    :param annotation: has format (annotation_str, kb_id, direct_ancestor_id)
    :type: tuple
    :param current_dict: output dict with docs and respective mentions already
        added
    :type current_dict
    :return: current_dict, updated with given annotation
    :rtype: dict

    >>> doc_id = 'DOC_1'
    >>> annotation = ('pneumonia', 'ID:02', 'ID:17')
    >>> current_dict = {'DOC_1': {'pneumonia': 'ID:02', 'ID:17'}}
    >>> add_or_ignore_annot(doc_id, annotation, current_dict)
    {'DOC_1':{'pneumonia':'ID:02','ID:17'}}
    >>> doc_id = 'DOC_2'
    >>> annotation = ('pneumonia', 'ID:02', 'ID:17')
    >>> current_dict = {'DOC_1':{'pneumonia':'ID:02','ID:17'}}
    >>> add_or_ignore_annot(doc_id, annotation, current_dict)
    {'DOC_1':{'pneumonia':'ID:02','ID:17'}, 'DOC_2':{'pneumonia':'ID:02','ID:17'}}
    """

    if doc_id in current_dict.keys(): 
        current_annotations = current_dict[doc_id]
        
        if annotation[0] not in current_annotations:
            # Only add 1 instance of a given entity mention per doc
            current_annotations[annotation[0]] = annotation[1:]
            current_dict[doc_id] = current_annotations
                          
    else:
        doc_dict = {annotation[0]: annotation[1:]}
        current_dict[doc_id] = doc_dict

    return current_dict


def parse_PBDMS(split, kb_data, partition):
    """Parses MeSH annotations in PubMed_DS corpus that are linked to concepts 
    of MEDIC, CTD-Chemicals, or CTD-Anatomy vocabularies (according to given
    kb_data, which already contains info about the target KB).
    
    :param split: PubMed DS split to parse
    :type split: str
    :param kb_data: represents a given knowledge base 
    :type kb_data: KnowledgeBase class object
    :param partition: medic, ctd-chem or ctd-anat
    :type: str
    :return: the dict output_PBDMS with format 
        {doc_id: {annotation_str: mesh_id, direct_ancestor}
    :rtype: dict
    
    """

    logging.info('Parsing PBDMS...')

    documents = list()
    split_filename = './data/corpora/pubmed_ds/split_' + split + '.txt'
            
    with open(split_filename, 'r', buffering=1, encoding='utf-8') \
            as input_split:
        documents = [doc for doc in input_split]
        input_split.close()

    output_PBDMS = dict()
    pbar = tqdm(total=len(documents))
    doc_count = int()
    
    for doc in documents: 
        parse_doc = False
        doc_count += 1

        if doc_count <= 30000 and partition == 'ctd_chem' or \
                partition == 'medic' or partition == 'ctd_anat':
                # ctd_chem: only parse some documents of split 0
            
            parse_doc = True
        
        if parse_doc:
            doc_dict = json.loads(doc)
            doc_id = doc_dict['_id']
            
            if len(doc_dict['mentions']) > 0:  
                
                for mention in doc_dict['mentions']:
                    mesh_id = 'MESH:' + mention['mesh_id'] 
                    
                    if mesh_id in kb_data.child_to_parent.keys():
                        direct_ancestor = kb_data.child_to_parent[mesh_id]
                        annotation = (mention['mention'], mesh_id, direct_ancestor)
                        output_PBDMS = add_or_ignore_annot(
                            doc_id, annotation, output_PBDMS)
        
        pbar.update(1)

    pbar.close()

    return output_PBDMS
   

def parse_CRAFT(kb_data):
    """Parses ChEBI or GO annotations in CRAFT corpus (according to given 
    kb_data).

    :param kb_data: repsresents a given knowledge base 
    :type kb_data: KnowledgeBase class object  
    :return: the dict output_CRAFT with format 
        {doc_id: {annotation_str: [KB_id, direct_ancestor]}
    :rtype: dict
    """

    logging.info('Parsing CRAFT corpus...')
    corpus_dir = './data/corpora/CRAFT-4.0.1/concept-annotation/'
        
    if kb_data.kb == 'chebi':
        corpus_dir += 'CHEBI/CHEBI/knowtator/'
        
    elif kb_data.kb == 'go_bp':
        corpus_dir += 'GO_BP/GO_BP/knowtator/'

    else:
        raise Exception('invalid target KB!')

    output_CRAFT = dict()
    documents = os.listdir(corpus_dir)
    pbar = tqdm(total=len(documents))

    for document in documents: 
        root = ET.parse(corpus_dir + document)
        doc_id = document.strip('.txt.knowtator.xml')
        annotations = dict()

        for annotation in root.iter('annotation'):
            annotation_id = annotation.find('mention').attrib['id']
            annotation_text = annotation.find('spannedText').text
            start_pos = annotation.find('span').attrib['start'],  
            end_pos = annotation.find('span').attrib['end']
            annotations[annotation_id] = [annotation_text, start_pos, end_pos] 
                
        for classMention in root.iter('classMention'):
            classMention_id = classMention.attrib['id']
            annotation_values = annotations[classMention_id]
            kb_id = classMention.find('mentionClass').attrib['id']
            annotation_str = annotation_values[0]
            
            if kb_id in kb_data.child_to_parent.keys():
                direct_ancestor = kb_data.child_to_parent[kb_id]
                annotation = (annotation_str, kb_id, direct_ancestor)
                output_CRAFT = add_or_ignore_annot(
                    doc_id, annotation, output_CRAFT)

        pbar.update(1)

    pbar.close()          
 
    return output_CRAFT


def parse_MedMentions(kb_data):
    """Parses UMLS annotations from MedMentions corpus and convert them to HP 
    annotations.
        
    :param kb_data: represents a given knowledge base 
    :type kb_data: KnowledgeBase class object  
    :return: output_MedMentions with format 
        {doc_id: {annotation_str: [HP_id, direct_ancestor]}
    :rtype: dict
    """
    
    logging.info('Parsing MedMentions corpus...')
    
    output_MedMentions = dict()
    filepath = './data/corpora/MedMentions/corpus_pubtator.txt'
    
    with open(filepath, 'r', buffering=1, encoding='utf-8') as corpus_file:

        for line in corpus_file:
            
            if '|t|' not in line \
                    and '|a|' not in line \
                    and line != '\n':
                
                doc_id = line.split('\t')[0]
                annotation_str = line.split('\t')[3]
                umls_id =line.split('\t')[5].strip('\n')
                
                if umls_id in kb_data.umls_to_hp.keys(): 
                    # UMLS concept has an equivalent HP concept
                    hp_id = kb_data.umls_to_hp[umls_id]

                    if hp_id in kb_data.child_to_parent.keys(): 
                        # Consider only HP concepts with 1 direct ancestor
                        direct_ancestor = kb_data.child_to_parent[hp_id].\
                            strip('\n')
                        annotation = (annotation_str, hp_id, direct_ancestor)
                        output_MedMentions = add_or_ignore_annot(
                            doc_id, annotation, output_MedMentions)
                            
    corpus_file.close()

    return output_MedMentions


def parse_annotations(partition, split=''):
    """Builds the output dict with annotations for given partition and split.
    
    :param partition: has value 'medic', 'ctd_anat', 'ctd_chem', 
        'chebi', 'go_bp' or 'hp'
    :type partition: str
    :param split: the dataset PBDMS is distributed among several splits, each 
        split is parsed sequentially (optional). Defaults to ''.
    :type split: str
    :return: out_dict with the annotations for specified partition/split, has 
        format {doc_id: {annotation_str: [kb_id, direct_ancestor_id]}}
    :rtype: dict
    """

    obo_list = ['hp', 'chebi', 'medic', 'go_bp'] 
    tsv_list = ['ctd_chem', 'ctd_anat']
    kb_data = kbs.KnowledgeBase(partition)
    out_dict = dict()

    if partition in obo_list:
        kb_data.load_obo(kb_data.kb)        

        if partition == 'hp':
            out_dict = parse_MedMentions(kb_data)

        elif partition == 'medic': 
            out_dict = parse_PBDMS(split, kb_data, partition)
            
        elif partition == 'chebi':
            out_dict = parse_CRAFT(kb_data)

        elif partition == 'go_bp':
            out_dict = parse_CRAFT(kb_data)

    elif partition in tsv_list:    
        kb_data.load_tsv(kb_data.kb)
       
        out_dict = parse_PBDMS(split, kb_data, partition)
    
    logging.info('Done!')
    return out_dict  