import logging
import json
import os
import xml.etree.ElementTree as ET
import sys
from tqdm import tqdm
from utils import check_if_annotation_is_valid

sys.path.append("./")

corpora_dir = 'data/corpora/'


def parse_Pubtator(dataset, dataset_dir, entity_type=None):
    """Gets annotations of given dataset in PubTator format.

    :param dataset: either 'chr', 'ncbi_disease', or 'bc5cdr' 
    :type dataset: str
    :param dataset_dir: directory where the files of the given dataset are
        located
    :type dataset: str
    :param entity_type: necessary only if parsing BC5CDR corpus, is either 
        "Chemical" or "Disease" (optional)
    :type dataset: str

    :return: annotations, with format {'doc_id': [annotation1, annotation2]}
    :rtype: dict
    """
    
    logging.info("Parsing annotations from the {} corpus".format(dataset))
    
    corpus_dir = corpora_dir + dataset_dir
    filenames = list()

    if dataset == 'chr':
        filenames = ['train.pubtator', 'dev.pubtator', 'test.pubtator']
    
    elif dataset == 'ncbi_disease':
        filenames = ['NCBItrainset_corpus.txt', 
                     'NCBIdevelopset_corpus.txt', 
                     'NCBItestset_corpus.txt']

    elif dataset == 'bc5cdr':
        filenames = ["CDR_TrainingSet.PubTator.txt", 
                     "CDR_DevelopmentSet.PubTator.txt",
                     "CDR_TestSet.PubTator.txt"]

    annotations = dict()
    
    for filename in filenames:
        
        with open(corpus_dir + filename, 'r') as corpus_file:
            data = corpus_file.readlines()
            corpus_file.close()
           
            for line in data:
                line_data = line.split("\t")
                doc_id = line_data[0]
                add_annot = True
                
                if len(line_data) == 6:
                    
                    kb_id = str()

                    if dataset == 'bc5cdr': 
                        
                        if line_data[4] == entity_type:
                            kb_id = "MESH_" + line_data[5].strip("\n")
                        
                        else:
                            add_annot = False
                    
                    elif dataset == 'ncbi_disease':

                        kb_id = line_data[5].strip("\n")

                        if kb_id[:4] != "OMIM" \
                            and '|' not in kb_id \
                            and '+' not in kb_id:
                            
                            # Do not consider OMIM or composite annotations
                            kb_id = "MESH_" + kb_id.strip(" ").strip("MESH:")
                        
                        else:
                            add_annot = False
                    
                    elif dataset == 'chr':
                        
                        kb_id = check_if_annotation_is_valid(line_data[5])

                        if kb_id == '':
                            add_annot = False

                    else:
                        
                        if '|' in line_data[5]:
                            kb_id = line_data[5].split('|')[1].strip('\n')  
                
                        else:
                            kb_id = line_data[5].strip('\n')

                        kb_id = kb_id.replace(':', '_')
                        
                    if add_annot:                    
                        annotation_text = line_data[3]
                        annotation = (kb_id, annotation_text)
                       
                        if doc_id in annotations.keys():
                            #current_values = annotations[doc_id]
                            #current_values.append(annotation)
                            #annotations[doc_id] = current_values
                            annotations[doc_id].append(annotation)
                        else:
                            annotations[doc_id] = [annotation]
    return annotations


def parse_GSC_corpus():
    """Gets HP annotations of the GSC+ corpus. 

    :return: annotations, with format {'doc_id': [annotation1, annotation2]}
    :rtype: dict
    """

    logging.info("Parsing annotations from the GSC+ corpus")
    corpus_dir = corpora_dir + "GSC+/Annotations/"
    annotations = dict() 

    for doc in os.listdir(corpus_dir):
        
        with open(corpus_dir + doc, 'r') as doc_file:
            data = doc_file.read()
            doc_file.close()
        
            data = data.split("\n")
            doc_annotations = list()
            
            for annot in data:
        
                if annot != '':
                    annot = annot.split("|")
                    annot_text = annot[1][1:]
                    hp_id = annot[0].split('\t')[1][:-1]
                    annotation = (hp_id, annot_text)
                    doc_annotations.append(annotation)
            
            annotations[doc] = doc_annotations

    return annotations


def parse_phaedra_corpus():

    annotations = dict()
    
    phaedra_dir = corpora_dir + "PHAEDRA_corpus/"
    subdirs = ["train/", "dev/", "test/"]

    for subdir in subdirs:
        
        subdir = phaedra_dir + subdir

        for doc in os.listdir(subdir):
            
            if doc[-2:] == "a1":
                doc_id = doc[:-3]

                with open(subdir + doc, "r") as file:
                    for line in file.readlines():
                        line_data = line.split("\t")
                        
                        if line_data[0][0] == "N":
                            kb_id = line_data[1].split(" ")[2]
                            
                            if kb_id[:4] == "MeSH":
                                entity_text = line_data[2].strip("\n")
                                mesh_id = kb_id.replace("MeSH:", "MESH_")
                                annotation = (mesh_id, entity_text)
                                
                                if doc_id in annotations.keys():
                                    annotations[doc_id].append(annotation)
                                
                                else:
                                    annotations[doc_id] = [annotation]
                    file.close()
    
    return annotations 


def parse_chebi_patents():
    
    chebi_patents_dir = corpora_dir + "ChebiPatents/"
    docs_list = os.listdir(chebi_patents_dir)
    annotations = dict()
    
    for file in (docs_list):
        
        tree = ET.parse(chebi_patents_dir + "/" + file + "/scrapbook.xml")
        root = tree.getroot()
        file_id = int(file[2:])

        for s in root.iter("snippet"):
            # get named entities
            for ne in s.findall("ne"):

                if ne.text is not None:
                    chebi_id = str()

                    if (not ne.get("chebi-id")
                        or ne.get("chebi-id").startswith("WO")
                        or "," in ne.get("chebi-id")):
                        chebi_id = "NIL"
                    
                    else:
                        chebi_id = ne.get("chebi-id")

                    annotation = (chebi_id, ne.text)

                    if file_id in annotations.keys():
                        annotations[file_id].append(annotation)
                    
                    else:
                        annotations[file_id] = [annotation]
    
    return annotations