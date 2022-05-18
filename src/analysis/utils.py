import logging
import os
import xml.etree.ElementTree as ET
import sys
from tqdm import tqdm
import csv
import logging
import obonet
import json
import os
import networkx as nx
import sys
sys.path.append("./")

#----------------------------------------------------------------------------
#                                     LOAD KBS
#----------------------------------------------------------------------------

root_dict = {"go_bp": ("GO_0008150", "biological_process"), 
            "chebi": ("CHEBI_00", "root"), 
            "hp": ("HP_0000001", "All"), 
            "medic": ("MESH_C", "Diseases"), 
            "ctd_anat": ("MESH_A", "Anatomy"), 
            "ctd_chem": ("MESH_D", "Chemicals")}

def load_obo(kb, only_graph=False):#, include_omim=False):
    """
    Loads MEDIC, HP, ChEBI from respective local obo file.

    :return: kb_graph, name_to_id, synonym_to_id, id_to_name 
    :rtype: Networkx MultiDiGraph object, dict, dict, id_to_name
    
    """
    
    logging.info("Loading " + kb)
    root_dict = {"go_bp": ("GO_0008150", "biological_process"), 
            "chebi": ("CHEBI_00", "root"), 
            "hp": ("HP_0000001", "All"), 
            "medic": ("MESH_C", "Diseases"), 
            "ctd_anat": ("MESH_A", "Anatomy")}

    kb_dir = '../../data/kb_files/'
    kb_filenames = {'medic': 'CTD_diseases.obo', 'hp': 'hp.obo',
                    'chebi': 'chebi.obo'}
 
    graph = obonet.read_obo(kb_dir + kb_filenames[kb]) 
    graph = graph.to_directed()
    
    # Create mappings
    name_to_id = {}
    synonym_to_id = {}
    #id_to_name = {}
    node_2_alt_ids = {}
    edge_list = []
    add_node = True

    for node in  graph.nodes(data=True):
        node_id, node_name = node[0], node[1]["name"]

        if 'alt_id' in node[1].keys():
            alt_ids = [alt_id.replace(':', '_') for alt_id in node[1]['alt_id'] if alt_id[:2] != 'DO']
            node_2_alt_ids[node_id_up] = alt_ids
           
        node_id_up = node_id.replace(':', '_')
        name_to_id[node_name] = node_id_up
        #id_to_name[node_id_up] = node_name

        if 'alt_id' in node[1].keys():
            alt_ids = [alt_id.replace(':', '_') for alt_id in node[1]['alt_id'] if alt_id[:2] != 'DO']
            node_2_alt_ids[node_id_up] = alt_ids
        
        if 'is_obsolete' in node[1].keys() and \
                node[1]['is_obsolete'] == True:
            add_node = False
            del name_to_id[node_name]
            del node_2_alt_ids[node_id_up]
            #del id_to_name[node_id] 
    
        if 'is_a' in node[1].keys() and add_node: 
            # The root node of the KB does not have is_a relationships 
            # with any ancestor
                
            for related_node in node[1]['is_a']: 
                # Build the edge_list with only "is-a" relationships
                
                related_node_up = related_node.replace(":", "_") 
                relationship = (node_id_up, related_node_up)
                
                edge_list.append(relationship) 
            
        if "synonym" in node[1].keys() and add_node: 
            # Check for synonyms for node (if they exist)
                
            for synonym in node[1]["synonym"]:
                synonym_name = synonym.split("\"")[1]
                synonym_to_id[synonym_name] = node_id_up
        
    # Create a MultiDiGraph object with only "is-a" relations 
    # this will allow the further calculation of shorthest path lenght
    kb_graph = nx.MultiDiGraph([edge for edge in edge_list])

    root_concept_name = root_dict[kb][1]
    root_id = root_dict[kb][0]

    if root_concept_name not in name_to_id.keys():
        root_id = root_dict[kb][0]
        name_to_id[root_concept_name] = root_id
        #id_to_name[root_id] = root_concept_name

    if kb == "chebi":
        # Add edges between the ontology root and sub-ontology roots
        chemical_entity = "CHEBI_24431"
        role = "CHEBI_50906"
        subatomic_particle = "CHEBI_36342"
        application = "CHEBI_33232"
        kb_graph.add_node(root_id, name="root")
        kb_graph.add_edge(chemical_entity, root_id, edgetype='is_a')
        kb_graph.add_edge(role, root_id, edgetype='is_a')
        kb_graph.add_edge(subatomic_particle, root_id, edgetype='is_a')
        kb_graph.add_edge(application, root_id, edgetype='is_a')
    
    #logging.info("Is kb_graph acyclic:", \
    #    nx.is_directed_acyclic_graph(kb_graph))
    logging.info("{} loading complete".format(kb))

    if only_graph==True:
        return kb_graph, root_id
    
    else:
        return kb_graph, name_to_id, synonym_to_id, node_2_alt_ids# id_to_name


def load_ctd_chemicals(only_graph=False):
    """
    Loads CTD-chemicals vocabulary from respective local tsv file.

    :return: kb_graph, name_to_id, synonym_to_id, id_to_name 
    :rtype: Networkx MultiDiGraph object, dict, dict, id_to_name

    """
    
    logging.info("Loading Chemical vocabulary")

    name_to_id = {}
    synonym_to_id = {}
    #id_to_name = dict()
    edge_list = []

    with open("../../data/kb_files/CTD_chemicals.tsv") as ctd_chem:
        reader = csv.reader(ctd_chem, delimiter="\t")
        row_count = 0
        
        for row in reader:
            row_count += 1
            
            if row_count >= 30:
                chemical_name = row[0] 
                chemical_id = "MESH_" + row[1][5:]
                chemical_parents = row[4].split('|')
                synonyms = row[7].split('|')
                name_to_id[chemical_name] = chemical_id
                #id_to_name[chemical_id] = chemical_name
                
                for parent in chemical_parents:
                    relationship = (chemical_id, parent[5:])
                    edge_list.append(relationship)
                
                for synonym in synonyms:
                    synonym_to_id[synonym] = chemical_id

    # Create a MultiDiGraph object with only "is-a" relations 
    # this will allow the further calculation of shorthest path lenght
    kb_graph = nx.MultiDiGraph([edge for edge in edge_list])
   
    root_concept_name = ("MESH_D", "Chemicals")
    root_id = ''

    if root_concept_name not in name_to_id.keys():
        root_id = root_dict['ctd_chem'][0]
        name_to_id[root_concept_name] = root_id
        #id_to_name[root_id] = root_concept_name
    
    if only_graph==False:
        return kb_graph, root_id
    
    else:
        return kb_graph, name_to_id, synonym_to_id, {}# id_to_name

#----------------------------------------------------------------------------
#                                  PARSE DATASETS
#----------------------------------------------------------------------------

corpora_dir = '../../data/corpora/'


def check_if_annotation_is_valid(annotation):
    print(annotation)
    output_kb_id = ''

    if '|' in annotation:
        # There are entities associated with two kb_ids, e.g.:
        # HMDB:HMDB01429|CHEBI:18367
        output_kb_id = annotation.split('|')[1].replace(':', '_')

        if output_kb_id[:4] == 'HMDB':
            output_kb_id = annotation.split('|')[0].replace(':', '_')
         
    else:
        
        if annotation[:5] == 'CHEBI':
            output_kb_id = annotation.replace(':', '_')
        
    output_kb_id = output_kb_id.strip('\n')

    # Check if output entity is valid
    assert output_kb_id[:5] == 'CHEBI' or output_kb_id == ''

    return output_kb_id


def parse_Pubtator(dataset, dataset_dir, entity_type=None):
    """Gets annotations of given dataset in PubTator format (test set).

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
    os.getcwd()
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

    train_annots = {}
    test_annots = {}
    annot = 0
    for filename in filenames:

        if 'train' in filename or 'Train' in filename:
            subset = 'train'
        
        elif 'dev' in filename or 'Dev' in filename:
            subset = 'dev'
        
        elif 'test' in filename or 'Test' in filename:
            subset = 'test'

        with open(corpus_dir + filename, 'r') as corpus_file:
            data = corpus_file.readlines()
            corpus_file.close()
           
            for line in data:
                line_data = line.split("\t")
                doc_id = line_data[0].split('|')[0]
                add_annot = False

                if len(line_data) == 6:

                    if dataset == 'bc5cdr' or dataset == 'ncbi_disease':
                        line_type = line_data[4]

                        if (line_type == entity_type and dataset == 'bc5cdr') or dataset =='ncbi_disease':
                            kb_id = line_data[5].strip("\n").strip(' ')
                            
                            if '+' in kb_id:
                                kb_id = kb_id.replace('+', '|')
                            
                            kb_ids = kb_id.split('|')
                            final_id = ''

                            for kb_id in kb_ids:
                                
                                if kb_id[0] == 'D' or kb_id[0] == 'C' or kb_id == '-1':
                                    kb_id = 'MESH_' + kb_id
                                
                                else:
                                    kb_id = kb_id.replace(':', '_')

                                final_id += kb_id + '|'
                            
                            final_id = final_id[:-1]
                            add_annot =True
              
                    elif dataset == 'chr':
                        
                        kb_id = check_if_annotation_is_valid(line_data[5])

                        if kb_id != '':# or kb_id not in kb_ids:
                            final_id = kb_id.replace(':', '_')
                            add_annot =True
                    
                    if add_annot:                    
                        annot_text = line_data[3]
                        annotation = (final_id, annot_text)
                        annot += 1

                        if subset == 'train' or subset == 'dev':
                                    
                            if doc_id in train_annots.keys():
                                train_annots[doc_id].append(annotation)
                            
                            else:
                                train_annots[doc_id] = [annotation]

                        elif subset == 'test':

                            if doc_id in test_annots.keys():
                                test_annots[doc_id].append(annotation)
                            
                            else:
                                test_annots[doc_id] = [annotation]
    
    return train_annots, test_annots


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
            #doc_annotations = {}
            
            for annot in data:
        
                if annot != '':
                    annot = annot.split("|")
                    annot_text = annot[1][1:]
                    hp_id = annot[0].split('\t')[1][:-1]

                    #if hp_id in kb_ids:
                    annotation = (hp_id, annot_text)
                    #doc_annotations[annot_text] = hp_id
                    if doc in annotations.keys():
                        annotations[doc].append(annotation)
                    
                    else:
                        annotations[doc] = [annotation]

            #annotations[doc] = doc_annotations

    return annotations


def parse_phaedra_corpus():

    train_annots = {}
    test_annots = {}
    
    phaedra_dir = corpora_dir + "PHAEDRA_corpus/"
    subdirs = ["train/", "dev/", "test/"]

    for subdir in subdirs:
        subset = subdir.strip('/')
        subdir = phaedra_dir + subdir

        for doc in os.listdir(subdir):
            
            if doc[-2:] == "a1":
                doc_id = doc[:-3]
                #doc_annots = {}

                with open(subdir + doc, "r") as file:
                    for line in file.readlines():
                        line_data = line.split("\t")
                        
                        if line_data[0][0] == "N":
                            kb_id = line_data[1].split(" ")[2]
                            
                            if kb_id[:4] == "MeSH":
                                entity_text = line_data[2].strip("\n")
                                mesh_id = kb_id.replace("MeSH:", "MESH_")
                                
                                #if mesh_id in kb_ids:
                                annotation = (mesh_id, entity_text)
                                #doc_annots[entity_text] = mesh_id
        
                                if subset == 'train' or subset == 'dev':

                                    if doc_id in train_annots.keys():
                                        train_annots[doc_id].append(annotation)
                                    
                                    else: 
                                        train_annots[doc_id] = [annotation]

                                elif subset == 'test':
                                    
                                    if doc_id in test_annots.keys():
                                        test_annots[doc_id].append(annotation)
                                    
                                    else: 
                                        test_annots[doc_id] = [annotation]

                file.close()

    return train_annots, test_annots


def import_input(filepath):
    """Import mentions from json file into dict"""

    #(kb_id, annotation_text)

    gold_standard = {}

    with open(filepath, 'r') as in_file:
        gold_standard = json.load(in_file)
        in_file.close()

    return gold_standard