import atexit
import csv
import logging
import obonet
import os
import networkx as nx
import pickle
import sys
sys.path.append("./")

root_dict = {"go_bp": ("GO_0008150", "biological_process"), 
            "chebi": ("CHEBI_00", "root"), 
            "hp": ("HP_0000001", "All"), 
            "medic": ("MESH_C", "Diseases"), 
            "ctd_anat": ("MESH_A", "Anatomy"), 
            "ctd_chem": ("MESH_D", "Chemicals")}

def load_obo(kb):
    """
    Loads MEDIC, HP, ChEBI from respective local obo file.

    :return: kb_graph, name_to_id, synonym_to_id, id_to_name 
    :rtype: Networkx MultiDiGraph object, dict, dict, id_to_name
    
    """
    
    logging.info("Loading " + kb)

    kb_dir = 'data/kb_files/'
    kb_filenames = {'medic': 'CTD_diseases.obo', 'hp': 'hp.obo',
                    'chebi': 'chebi.obo'}
 
    graph = obonet.read_obo(kb_dir + kb_filenames[kb]) 
    graph = graph.to_directed()
    
    # Create mappings
    name_to_id = dict()
    synonym_to_id = dict()
    id_to_name = dict()
    edge_list = list()
    add_node = True

    for node in  graph.nodes(data=True):
        node_id, node_name = node[0], node[1]["name"]
        
        if node_id[0:4] != "OMIM":
                
            node_id_up = node_id.replace(':', '_')
            name_to_id[node_name] = node_id_up
            id_to_name[node_id_up] = node_name

            if 'is_obsolete' in node[1].keys() and \
                    node[1]['is_obsolete'] == True:
                add_node = False
                del name_to_id[node_name]
                del id_to_name[node_id] 
        
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
        id_to_name[root_id] = root_concept_name

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

    return kb_graph, name_to_id, synonym_to_id, id_to_name


def load_ctd_chemicals():
    """
    Loads CTD-chemicals vocabulary from respective local tsv file.

    :return: kb_graph, name_to_id, synonym_to_id, id_to_name 
    :rtype: Networkx MultiDiGraph object, dict, dict, id_to_name

    """
    
    logging.info("Loading Chemical vocabulary")

    name_to_id = dict()
    synonym_to_id = dict()
    id_to_name = dict()
    edge_list = list()

    with open("data/kb_files/CTD_chemicals.tsv") as ctd_chem:
        reader = csv.reader(ctd_chem, delimiter="\t")
        row_count = int()
        
        for row in reader:
            row_count += 1
            
            if row_count >= 30:
                chemical_name = row[0] 
                chemical_id = "MESH_" + row[1][5:]
                chemical_parents = row[4].split('|')
                synonyms = row[7].split('|')
                name_to_id[chemical_name] = chemical_id
                id_to_name[chemical_id] = chemical_name
                
                for parent in chemical_parents:
                    relationship = (chemical_id, parent[5:])
                    edge_list.append(relationship)
                
                for synonym in synonyms:
                    synonym_to_id[synonym] = chemical_id

    # Create a MultiDiGraph object with only "is-a" relations 
    # this will allow the further calculation of shorthest path lenght
    kb_graph = nx.MultiDiGraph([edge for edge in edge_list])
   
    root_concept_name = root_dict['ctd_chem'][1]
    root_id = str()

    if root_concept_name not in name_to_id.keys():
        root_id = root_dict['ctd_chem'][0]
        name_to_id[root_concept_name] = root_id
        id_to_name[root_id] = root_concept_name
    
    return kb_graph, name_to_id, synonym_to_id, id_to_name

