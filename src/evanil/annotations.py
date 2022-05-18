import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
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
    is_in_doc = False

    if doc_id in current_dict.keys(): 
        current_annotations = current_dict[doc_id]
        
        if annotation[0] not in current_annotations:
            # Only add 1 instance of a given entity mention per doc
            current_annotations[annotation[0]] = annotation[1:]
            current_dict[doc_id] = current_annotations
        
        else:
            is_in_doc = True
                          
    else:
        doc_dict = {annotation[0]: annotation[1:]}
        current_dict[doc_id] = doc_dict

    return current_dict, is_in_doc


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

    documents = []
    split_filename = 'data/corpora/pubmed_ds/split_' + split + '.txt'
            
    with open(split_filename, 'r', buffering=1, encoding='utf-8') \
            as input_split:
        documents = [doc for doc in input_split]
        input_split.close()

    entity_types = {'ctd_chem': 'Chemical', 'medic': 'Disease', 'ctd_anat': 'Anatomical'}
    
    pbar = tqdm(total=len(documents))
    pubtator_output = ''
    out_dict = {}
    
    for i, doc in enumerate(documents): 
        parse_doc = False
        
        if (i <= 30000 and partition == 'ctd_chem') or \
                (partition == 'medic') or (partition == 'ctd_anat'):
                # ctd_chem: only parse some documents of split 0
            
            parse_doc = True
        
        if parse_doc:
            doc_annotations_txt = []
            doc_annotations = []
            doc_dict = json.loads(doc)
            doc_id = doc_dict['_id']
            
            if len(doc_dict['mentions']) > 0:  
                
                for mention in doc_dict['mentions']:
                    mesh_id = 'MESH:' + mention['mesh_id'] 
                    text = mention['mention']

                    if text not in doc_annotations_txt:
                        
                        if mesh_id in kb_data.child_to_parent.keys():
                            direct_ancestor = kb_data.child_to_parent[mesh_id]
                            start = mention['start_offset']
                            end = mention['end_offset']
                            
                            # Pubtator format
                            annotation_out = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                                doc_id, start, end, text, entity_types[partition], direct_ancestor)
                            
                            doc_annotations.append(annotation_out)
                            doc_annotations_txt.append(text)

                            # Output dictionary
                            annotation = (text, mesh_id, direct_ancestor)
                            out_dict, is_in_doc = add_or_ignore_annot(
                            doc_id, annotation, out_dict)

                
                if len(doc_annotations_txt) >= 1:
                    #add present doc info to dataset
                    title = doc_dict['title'].strip('[')
                    title = title.strip('].')
                    abs_start_pos = len(doc_dict['title']) + 1
                    abstract = doc_dict['text'][abs_start_pos:]

                    pubtator_output += '{}|t|{}'.format(doc_id, title) + '\n'
                    pubtator_output += '{}|a|{}'.format(doc_id, abstract) + '\n'

                    for annot in doc_annotations:
                        pubtator_output += annot
                    
                    pubtator_output += '\n'

        pbar.update(1)

    pbar.close()
    
    return out_dict, pubtator_output[:-1]
   

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
    corpus_dir = 'data/corpora/CRAFT-4.0.1/'
    
    annotations_dir = corpus_dir + 'concept-annotation/'
        
    if kb_data.kb == 'chebi':
        annotations_dir += 'CHEBI/CHEBI/knowtator/'
        entity_type = 'Chemical'
        
    elif kb_data.kb == 'go_bp':
        annotations_dir += 'GO_BP/GO_BP/knowtator/'
        entity_type = 'Bioprocess'

    else:
        raise Exception('invalid target KB!')

    out_dict = {}
    documents = os.listdir(annotations_dir)
    pbar = tqdm(total=len(documents))
    txt_dir = corpus_dir + 'articles/txt/'
    pubtator_output = ''

    for document in documents: 
        root = ET.parse(annotations_dir + document)
        doc_id = document.strip('.txt.knowtator.xml')
        
        #Get text from file
        with open(txt_dir + doc_id + '.txt', 'r') as txt_file:
            txt = txt_file.read()
            txt_file.close()
        
        data = txt.split('\n')
        title = data[0]
        txt_up = ''

        for i, elem in enumerate(data[1:]):
            
            if i != 0:

                if elem == '':
                    txt_up += '\t'

                else:
                    txt_up += elem
        
        pubtator_output += '{}|t|{}'.format(doc_id, title) + '\n'
        pubtator_output += '{}|a|{}'.format(doc_id, txt_up) + '\n'
        
        annotations = {}
        doc_annotations_txt = []

        for annotation in root.iter('annotation'):
            annotation_id = str(annotation.find('mention').attrib['id'])
            annotation_text = str(annotation.find('spannedText').text)
            start_pos = int(annotation.find('span').attrib['start'])  
            end_pos = int(annotation.find('span').attrib['end'])
            annotations[annotation_id] = [annotation_text, start_pos, end_pos] 
                
        for classMention in root.iter('classMention'):
            classMention_id = classMention.attrib['id']
            annotation_values = annotations[classMention_id]
            kb_id = classMention.find('mentionClass').attrib['id']
            text = annotation_values[0]

            if text not in doc_annotations_txt:
            
                if kb_id in kb_data.child_to_parent.keys():
                    direct_ancestor = kb_data.child_to_parent[kb_id]
                    start = annotation_values[1]
                    end = annotation_values[2]
                    
                    # Pubtator format
                    annotation_out = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        doc_id, start, end, text, entity_type, direct_ancestor)
                    pubtator_output += annotation_out
                    doc_annotations_txt.append(text)

                    # Output dictionary
                    annotation = (text, kb_id, direct_ancestor)
                    out_dict, is_in_doc = add_or_ignore_annot(
                    doc_id, annotation, out_dict)

        pubtator_output += '\n\n'

        pbar.update(1)

    pbar.close()          
    
    return out_dict, pubtator_output[:-1]


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
    
    out_dict= {}
    pubtator_output = ''
    filepath = './data/corpora/MedMentions/corpus_pubtator.txt'
    
    with open(filepath, 'r', buffering=1, encoding='utf-8') as corpus_file:

        for line in corpus_file:
            
            if '|t|' not in line \
                    and '|a|' not in line \
                    and line != '\n':
                
                doc_id = line.split('\t')[0]
                annot_text = line.split('\t')[3]
                umls_id =line.split('\t')[5].strip('\n')
                start = line.split('\t')[1]
                end = line.split('\t')[2]
                
                if umls_id in kb_data.umls_to_hp.keys(): 
                    # UMLS concept has an equivalent HP concept
                    hp_id = kb_data.umls_to_hp[umls_id]

                    if hp_id in kb_data.child_to_parent.keys(): 
                        # Consider only HP concepts with 1 direct ancestor
                        direct_ancestor = kb_data.child_to_parent[hp_id].\
                            strip('\n')
                        annotation = (annot_text, hp_id, direct_ancestor)
                        out_dict, is_in_doc = add_or_ignore_annot(
                            doc_id, annotation, out_dict)

                        if not is_in_doc:
                            annotation_pub = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                                doc_id, start, end, annot_text, 'Disease', direct_ancestor)
                            pubtator_output += annotation_pub + '\n'

            elif '|t|' in line or '|a|' in line:
                pubtator_output += line + '\n'

    corpus_file.close()

    return out_dict, pubtator_output


def parse_annotations(kb_data, partition, split=''):
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
    pubtator_output = ''

    if partition == 'hp':
        out_dict, pubtator_output = parse_MedMentions(kb_data)

    elif partition == 'medic' or partition == 'ctd_chem' or partition == 'ctd_anat': 
        out_dict, pubtator_output = parse_PBDMS(split, kb_data, partition)
    
    elif partition == 'chebi' or partition == 'go_bp':
        out_dict, pubtator_output = parse_CRAFT(kb_data)      
    
    logging.info('Done!')
    return out_dict, pubtator_output
