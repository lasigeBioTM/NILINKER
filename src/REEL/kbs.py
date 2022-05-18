import csv
import logging
import obonet
import os
import networkx as nx
import sys
sys.path.append("./")

root_dict = {"go_bp": ("GO_0008150", "biological_process"), 
            "chebi": ("CHEBI_00", "root"), 
            "hp": ("HP_0000001", "All"), 
            "medic": ("MESH_C", "Diseases"), 
            "ctd_anat": ("MESH_A", "Anatomy"), 
            "ctd_chem": ("MESH_D", "Chemicals")}


def load_obo(kb, only_graph=False):#, include_omim=False):
    """
    Loads MEDIC, HP, ChEBI from respective local obo file.

    :return: kb_graph, name_to_id, synonym_to_id, node_2_alt_ids 
    :rtype: Networkx MultiDiGraph object, dict, dict, dict
    
    """
    
    logging.info("Loading " + kb)
    root_dict = {"go_bp": ("GO_0008150", "biological_process"), 
            "chebi": ("CHEBI_00", "root"), 
            "hp": ("HP_0000001", "All"), 
            "medic": ("MESH_C", "Diseases"), 
            "ctd_anat": ("MESH_A", "Anatomy")}

    kb_dir = 'data/kb_files/'
    kb_filenames = {'medic': 'CTD_diseases.obo', 'hp': 'hp.obo',
                    'chebi': 'chebi.obo'}
 
    graph = obonet.read_obo(kb_dir + kb_filenames[kb]) 
    graph = graph.to_directed()
    
    # Create mappings
    name_to_id = {}
    synonym_to_id = {}
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

        if 'alt_id' in node[1].keys():
            alt_ids = [alt_id.replace(':', '_') for alt_id in node[1]['alt_id'] if alt_id[:2] != 'DO']
            node_2_alt_ids[node_id_up] = alt_ids
        
        if 'is_obsolete' in node[1].keys() and \
                node[1]['is_obsolete'] == True:
            add_node = False
            del name_to_id[node_name]
            del node_2_alt_ids[node_id_up]
    
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
    root_id = str()

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
    
    logging.info("{} loading complete".format(kb))

    if only_graph==True:
        return kb_graph
    
    else:
        return kb_graph, name_to_id, synonym_to_id, node_2_alt_ids


def load_ctd_chemicals(only_graph=False):
    """
    Loads CTD-chemicals vocabulary from respective local tsv file.

    :return: kb_graph, name_to_id, synonym_to_id, {}
    :rtype: Networkx MultiDiGraph object, dict, dict, dict

    """
    
    logging.info("Loading Chemical vocabulary")

    name_to_id = {}
    synonym_to_id = {}
    edge_list = []

    with open("data/kb_files/CTD_chemicals.tsv") as ctd_chem:
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
    
    if only_graph==False:
        return kb_graph
    
    else:
        return kb_graph, name_to_id, synonym_to_id, {}

"""
def load_txt(kb):
    \"""MEDIC (6 Jul 2012) or CTD-Chemicals (04 NOV 2019)\"""

    # First import concept list from the old version
    if kb == 'medic':
        filepath = 'data/kb_files/medic_06Jul2012.txt'

    elif kb == 'ctd_chem':
        filepath = 'data/kb_files/ctd_chemical_04Nov2019.txt'

    with open(filepath, 'r') as kb_file:
        data = kb_file.readlines()  
        kb_file.close()

    name_to_id = {}
    synonym_to_id = {}
    node_2_alt_ids = {}
    
    for line in data:
        line_data = line.split('|')
        first_id = ''
        second_id = ''
        synonyms = []
        
        if line_data[0][0] == 'C' or line_data[0][0] == 'D':
            first_id = 'MESH_' + line_data[0].strip('\n')
            
            if line_data[1] != '':
                second_id = 'OMIM_' + line_data[1].strip('\n')
            
        
        else:
            first_id = 'OMIM_' + line_data[0].strip('\n')

            if line_data[1] != '':
                second_id = 'MESH_' + line_data[1].strip('\n')
            
     
        concept_name = line_data[2].strip('\n')
        synonyms = line_data[3:]
        name_to_id[concept_name] = first_id

        for synonym in synonyms:
            
            if synonym != '':
                synonym_to_id[synonym.strip('\n')] = first_id


        if second_id != '':
            node_2_alt_ids[first_id] = second_id

    
    # Import current version of the ontology in order to obtain the edge list 
    # generate the ontology graph for later use
    
    if kb == 'medic':
        kb_graph = load_obo('medic', only_graph=True)
    
    elif kb == 'ctd_chem':
        kb_graph = load_ctd_chemicals('ctd_chem', only_graph=True)
    
    return kb_graph, name_to_id, synonym_to_id, node_2_alt_ids
"""