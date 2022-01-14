# -*- coding: utf-8 -*-
from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from utils import check_if_annotation_is_valid
from tqdm import tqdm
import os
import sys
import xml.etree.ElementTree as ET
import json

sys.path.append("./")


def import_chr_relations():
    """Gets the relations (ChEBI-ChEBI) described in the CHR corpus.
    
    :return: extracted_relations, with format 
        {"chebi_id1": [chebi_id2, chebi_id3]}
    :rtype: dict
    """

    extracted_relations = dict()
    corpus_dir = 'data/corpora/CHR_corpus/'
    filenames = ['train.pubtator', 'dev.pubtator', 'test.pubtator']

    for filename in filenames:

        with open(corpus_dir + filename, 'r') as corpus_file:
            data = corpus_file.readlines()
            corpus_file.close()
           
            for line in data:
                line_data = line.split("\t")
                add_relation = True
                
                if len(line_data) == 4:

                    annotation1 = line_data[2]
                    annotation2 = line_data[3]

                    entity1 = check_if_annotation_is_valid(annotation1)
                    entity2 = check_if_annotation_is_valid(annotation2)
                    
                    if entity1 != '' and entity2 != '':
                        
                        if entity1 in extracted_relations.keys():
                            extracted_relations[entity1].append(entity2) 
                        
                        else:
                            extracted_relations[entity1] = [entity2]

                        if entity2 in extracted_relations.keys():
                            extracted_relations[entity2].append(entity1) 
                        
                        else:
                            extracted_relations[entity2] = [entity1]
    

    return extracted_relations


def create_hp_relations():
    """Builds dict with disease-disease relations from the HPO file 
    'phenotype_to_genes.txt'.
    
    :return: extracted_relations, with format 
        {"hp_id1": [hp_id2, hp_id3, ...]}
    :rtype: dict"""
    
    annotations_path = "data/kb_files/phenotype_to_genes.txt" 

    with open(annotations_path, 'r') as annotations_file:
            data = annotations_file.readlines()
            annotations_file.close()

    extracted_relations_temp  = dict()
    line_count = 0

    for line in data:

        if line_count > 1:
            hp_id = line.split("\t")[0].replace(":","_")
            gene_id = line.split("\t")[2]

            if gene_id in extracted_relations_temp.keys():
                extracted_relations_temp[gene_id].append(hp_id)
                    
            else:
                extracted_relations_temp[gene_id] = [hp_id]
            
        line_count += 1
    
    extracted_relations = dict()
    
    pbar = tqdm(total=len(extracted_relations_temp.keys()))

    for key in extracted_relations_temp.keys():
        entities = extracted_relations_temp[key]
        
        for entity_1 in entities:
            
            for entity_2 in entities:            

                if entity_1 != entity_2:

                    if entity_1 in extracted_relations.keys():
                        current = extracted_relations[entity_1]
                            
                        if entity_2 not in current:
                            current.append(entity_2)
                            extracted_relations[entity_1] = current
                        
                    elif entity_1 not in extracted_relations.keys():
                        extracted_relations[entity_1] = [entity_2]
                        
                    if entity_2 in extracted_relations.keys():
                        current = extracted_relations[entity_2]
                            
                        if entity_1 not in current:
                            current.append(entity_1)
                            extracted_relations[entity_2] = current
                        
                    elif entity_2 not in extracted_relations.keys():
                        extracted_relations[entity_2] = [entity_1]
        pbar.update(1)
    
    pbar.close()
    
    output = json.dumps(extracted_relations)

    with open("data/kb_files/hp_relations.json", "w") as outfile:
        outfile.write(output)
        outfile.close()


def import_hp_relations():
    """Import HP relations (disease-disease) from the previously built dict."""

    datafilepath = "data/kb_files/hp_relations.json"

    with open(datafilepath, "r") as datafile:
        relations = json.load(datafile) 
    
    return relations


#-----------------------------------------------------------------------------
#                   Import relations from the BC5CDR corpus 
#                   (disease-disease or chemical-chemical)
#-----------------------------------------------------------------------------

def import_cdr_relations_pubtator(dataset, subset):
    """
    Imports chemical-disease interactions from BC5CDR corpus in PubTator format 
    into dict.

    :param dataset: the target dataset
    :type dataset: str
    :param subset:  either "train", "dev", "test" or "all"
    :type subset: str
    
    :return: extracted_relations, with format 
        {"disease_id1": [disease_id2, disease_id3]}
    :rtype: dict
    """

    corpus_dir = 'data/corpora/BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/'
    filenames = list()
    extracted_relations = dict()
    extracted_relations_temp  = dict()
    
    if subset == "train":
        filenames.append("CDR_TrainingSet.PubTator.txt")
    
    elif subset == "dev":
        filenames.append("CDR_DevelopmentSet.PubTator.txt")
    
    elif subset == "test":
        filenames.append("CDR_TestSet.PubTator.txt")
    
    elif subset == "all":
        filenames.append("CDR_TrainingSet.PubTator.txt")
        filenames.append("CDR_DevelopmentSet.PubTator.txt")
        filenames.append("CDR_TestSet.PubTator.txt")
  
    for filename in filenames:
        
        with open(corpus_dir + filename, 'r') as corpus_file:
            data = corpus_file.readlines()
            corpus_file.close()
           
            for line in data:
                line_data = line.split("\t")
                
                if len(line_data) == 4 and line_data[1] == "CID":
                    # Chemical-disease Relation 
                    chemical_id = 'MESH_' + line_data[2]
                    disease_id = 'MESH_' + line_data[3].strip("\n")

                    if dataset == "bc5cdr_medic":
                        # We want to get the disease-disease relations
                        
                        if chemical_id in extracted_relations_temp.keys():
                            added = extracted_relations_temp[chemical_id]
                            added.append(disease_id) 
                            extracted_relations_temp[chemical_id] = added
                                
                        else:
                            extracted_relations_temp[chemical_id] = [disease_id]

                    elif dataset == "bc5cdr_chem":
                        # We want to get the chemical-chemical relations
                        
                        if disease_id in extracted_relations_temp.keys():
                            added = extracted_relations_temp[disease_id]
                            added.append(chemical_id) 
                            extracted_relations_temp[disease_id] = added
                                
                        else:
                            extracted_relations_temp[disease_id] = [chemical_id]

    # Two disease terms are related if associated with the same chemical
    # Two chemical terms are related if associated with the same disease
    
    for key in extracted_relations_temp.keys():
        entities = extracted_relations_temp[key]
        
        for entity_1 in entities:
            
            for entity_2 in entities:            

                if entity_1 != entity_2:

                    if entity_1 in extracted_relations.keys():
                        current = extracted_relations[entity_1]
                            
                        if entity_2 not in current:
                            current.append(entity_2)
                            extracted_relations[entity_1] = current
                        
                    elif entity_1 not in extracted_relations.keys():
                        extracted_relations[entity_1] = [entity_2]
                        
                    if entity_2 in extracted_relations.keys():
                        current = extracted_relations[entity_2]
                            
                        if entity_1 not in current:
                            current.append(entity_1)
                            extracted_relations[entity_2] = current
                        
                    elif entity_2 not in extracted_relations.keys():
                        extracted_relations[entity_2] = [entity_1]
    
    return extracted_relations
                    

def import_bolstm_output():
    """
    Parses the BO-LSTM output to obtain extracted relationship between ChEBI 
    entities in the ChEBI corpus.

    :return: extracted_relations, with format 
        {"disease_id1": [disease_id2, disease_id3]}
    :rtype: dict
    
    """
    
    id_to_kb_id = dict()

    # Create mappings between entity id and ChEBI id in the original file 
    # (pre-bolstm)
    path = "full_model_temp.chebicraftresults.txt"
    corpus_dir = "converted_chebi_craft_corpus/"
    docs_list = os.listdir(corpus_dir)
    
    for document in docs_list:
        tree = ET.parse(corpus_dir + document)
        root = tree.getroot()

        for entity in root.iter("entity"):
            kb_id = entity.get("ontology_id")
            entity_id = entity.get("id")
            id_to_kb_id[entity_id] = kb_id

    # Read the bolstm output file and import relations
    bolstm_file = open(path, 'r')

    extracted_relations = dict()
    
    for line in bolstm_file.readlines():

        if line[:7] != "entity1": 
            # Ignore the file header

            entity1_id = line.split('\t')[0]
            entity2_id = line.split('\t')[1]
            
            if line.split('\t')[2] == "effect\n": 
                # There is a relation between the two entities
                # Update the entity id with the respective KB id
                entity1_kb_id = id_to_kb_id[entity1_id]  
                entity2_kb_id = id_to_kb_id[entity2_id]

                if entity1_kb_id in extracted_relations.keys():

                    if entity2_kb_id not in extracted_relations[entity1_kb_id]:
                        old_values = extracted_relations[entity1_kb_id]
                        old_values.append(entity2_kb_id) 
                        extracted_relations[entity1_kb_id] = old_values
                            
                else:
                    new_values = [entity2_kb_id]
                    extracted_relations[entity1_kb_id] = new_values

                
                if entity2_kb_id in extracted_relations.keys():
                    
                    if entity1_kb_id not in extracted_relations[entity2_kb_id]:
                        old_values = extracted_relations[entity2_kb_id]
                        old_values.append(entity1_kb_id) 
                        extracted_relations[entity2_kb_id] = old_values
                
                else:
                    new_values = [entity1_kb_id]
                    extracted_relations[entity2_kb_id] = new_values
                    
    bolstm_file.close()

    return extracted_relations


def craft_input_to_bolstm():
    """Converts the documents in the CRAFT corpus to the input structure 
       of BO-LSTM."""

    # Sentence segmentation using Spacy
    nlp = English()
    sentencizer = Sentencizer()
    nlp.add_pipe(sentencizer)

    # Parse each document in corpus directory -
    corpus_dir = "chebi_craft_corpus/"
    docs_list = os.listdir(corpus_dir)

    for idoc, file in enumerate(docs_list):
                
        if file[-3:] == "xmi":
            file_path = corpus_dir + file 
            file_id = str(file[:-4])
            
            #Retrieve the entire document text
            tree = ET.parse(file_path)
            root = tree.getroot()

            for child in root: 
                
                if child.tag == "{http:///uima/cas.ecore}Sofa":
                    document_text = child.attrib["sofaString"]

            # Import annotations from annotations file into annotation_list
            annotation_list = []
            annotation_file = open(file_path[:-3] + "ann", "r")

            for line in annotation_file.readlines():
                entity_text = line.split("\t")[2].strip("\n")
                kb_id = line.split("\t")[1].split(" ")[0].replace("_", ":")
                offset_begin = int(line.split("\t")[1].split(" ")[1])
                offset_end = int(line.split("\t")[1].split(" ")[2].\
                    split(";")[0])
                annotation_list.append((entity_text, kb_id, offset_begin, 
                                        offset_end))

            annotation_file.close()

            # Create the xml tree for output file
            new_root = ET.Element("document") 
            new_root.set("id", file_id)

            # Iterate over each sentence in document
            docSpacy = nlp(document_text) 
            sentence_count, token_count = 0, 0

            for sentence in docSpacy.sents:
                sentence_count += 1
                begin_offset = token_count + 1
                token_count += len(sentence.text) + 1
                final_offset = token_count
                sentence_id = str(file_id) + ".s" + str(sentence_count)
                entity_count = 0
                entity_check = []

                # Create xml structure for sentence
                new_sentence = ET.SubElement(new_root, "sentence")
                new_sentence.set("id", sentence_id)
                new_sentence.set("text", sentence.text)
                
                # Check if there is any annotation present in the current 
                # sentence
                valid_entities_list = []
                
                for annotation in annotation_list:
                        
                    if annotation[2] >= begin_offset \
                            and annotation[2] <= final_offset: 
                        # There is an annotation in this sentence
                        entity_text = annotation[0]
                        
                        if entity_text not in entity_check: 
                            # The entity was not added to sentence

                            #Upgrade the entity offset in sentence context
                            entity_begin_offset = sentence.text.find(
                                entity_text)
                            
                            if entity_begin_offset > -1: 
                                entity_count += 1
                                entity_id = sentence_id + ".e" \
                                    + str(entity_count)
                                entity_final_offset = entity_begin_offset \
                                    + len(entity_text) - 1
                                entity_offset = str(entity_begin_offset) \
                                    + "-" + str(entity_final_offset)
                                entity_check.append(entity_text)
                                valid_entities_list.append(entity_id)
                                
                                # Create xml structure for annotation
                                new_entity = ET.SubElement(new_sentence, 
                                                           "entity")
                                new_entity.set("id", entity_id)
                                new_entity.set("charOffset", entity_offset)
                                new_entity.set("type", "chebi")
                                new_entity.set("text", entity_text)
                                new_entity.set("ontology_id", annotation[1])

                # Create Xml structure for pairs of entities in sentence
                pair_count = 0
                pair_check = []

                for valid_entity in valid_entities_list:

                    for valid_entity_2 in valid_entities_list:
                        print(valid_entity)
                        if valid_entity != valid_entity_2: 
                            # Create a pair between two different entities
                            pair_check_id1 = valid_entity + "_" \
                                + valid_entity_2
                            pair_check_id2 = valid_entity_2 + "_" \
                                + valid_entity

                            if pair_check_id1 not in pair_check \
                                    and pair_check_id2 not in pair_check : 
                                # Prevent duplicate pairs
                                pair_count += 1
                                pair_id = sentence_id + ".p" + str(pair_count)
                                pair_check.append(pair_check_id1)
                                pair_check.append(pair_check_id2)

                                new_pair = ET.SubElement(new_sentence, "pair")
                                new_pair.set("id", pair_id)
                                new_pair.set("e1", valid_entity), new_pair.\
                                    set("e2", valid_entity_2)
                                new_pair.set("ddi", "false")

            #Create an .xml output file
            ET.ElementTree(new_root).write("./bolstm/converted_chebi_craft/" \
                + file_id + ".xml", xml_declaration=True)


#-----------------------------------------------------------------------------
#        Import relations from the PHAEDRA corpus (chemical-chemical)
#-----------------------------------------------------------------------------

def create_term_dictionary(doc_filepath):
    """[Only applicable to the PHAEDRA corpus]"""

    text_to_mesh_id = dict()
    text_to_term = dict()
    term_to_mesh_id = dict()
    
    with open(doc_filepath.replace("a2", "a1"), "r") as file:
        for line in file.readlines():
            line_data = line.split("\t")
            
            if line_data[0][0] == "N":
                kb_id = line_data[1].split(" ")[2]
                
                if kb_id[:4] == "MeSH":
                    entity_text = line_data[2].strip("\n")
                    mesh_id = kb_id.replace("MeSH:", "MESH_")
                    text_to_mesh_id[entity_text] = mesh_id

            if line_data[0][0] == "T":
                term = line_data[0]
                type = line_data[1].split(" ")[0]

                if type == "Pharmacological_substance":
                    text = line_data[2].strip("\n")
                    text_to_term[text] = term

    # Only include in the output the terms associated with pharmacological 
    # substances
    for text in text_to_term.keys():
        term =  text_to_term[text]

        if text in text_to_mesh_id.keys():
            mesh_id = text_to_mesh_id[text]

            term_to_mesh_id[term] = mesh_id
        
        else:
            term_to_mesh_id[term] = "NIL"

    return term_to_mesh_id


def get_relation_between_terms(terms, term_to_mesh_id, extracted_relations):
    """[Only applicable to the PHAEDRA corpus] Converts a given list of term 
    ids into mesh ids, if available, and then update the extracted_relations 
    dict with all the relations between the 
    terms."""

    relation_mesh_ids = list()

    # Check if the terms have a mesh id
    for term in terms:
        
        if term in term_to_mesh_id.keys():
            mesh_id = term_to_mesh_id[term]
            
            if mesh_id != "NIL":
                relation_mesh_ids.append(mesh_id)

    # Get the relation
    if len(relation_mesh_ids) > 1:

        for mesh_id in relation_mesh_ids:

            for other_mesh_id in relation_mesh_ids:

                if mesh_id != other_mesh_id:

                    if mesh_id in extracted_relations.keys():
                        current = extracted_relations[mesh_id]

                        if other_mesh_id not in current:
                            extracted_relations[mesh_id].append(other_mesh_id)

                    else:
                        extracted_relations[mesh_id] = [other_mesh_id]

    return extracted_relations


def import_phaedra_relations():
    
    extracted_relations = dict()

    phaedra_dir = "data/corpora/PHAEDRA_corpus/"
    subdirs = ["train/", "dev/", "test/"]

    for subdir in subdirs:
        
        subdir = phaedra_dir + subdir

        for doc in os.listdir(subdir):
            
            if doc[-2:] == "a2":
                doc_id = doc[:-3]
                filepath = subdir + doc
                term_to_mesh_id = create_term_dictionary(filepath)
                
                with open(filepath, "r") as file:
                    
                    for line in file.readlines():
                        
                        if line[0] == "E":
                            # Event
                            data = line.split("\t")
                            terms_tmp = data[1].split(" ")[1:]
                            terms = list()

                            for term in terms_tmp:
                                term = term.strip("\n").split(":")[1]
                                terms.append(term)
                            
                            extracted_relations = \
                                get_relation_between_terms(terms, 
                                                           term_to_mesh_id, 
                                                           extracted_relations)
                                
                        elif line[0] == "R":
                            # Relation
                            data = line.split("\t")[1].split(" ")
                            
                            terms = [data[1].strip("\n").split(":")[1],
                                     data[2].strip("\n").split(":")[1]]

                            extracted_relations = \
                                get_relation_between_terms(terms, 
                                                           term_to_mesh_id, 
                                                           extracted_relations)
    
    return extracted_relations