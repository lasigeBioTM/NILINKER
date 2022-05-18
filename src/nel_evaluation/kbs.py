import csv
import obonet
import sys


def load_obo(kb):
    """
    Loads MEDIC, HP, ChEBI from respective local obo file.

    :return: kb_graph, name_to_id, synonym_to_id, id_to_name 
    :rtype: Networkx MultiDiGraph object, dict, dict, id_to_name
    
    """
    
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
    
    names = []

    for node in  graph.nodes(data=True):
        names.append(node[1]["name"])
        
        if "synonym" in node[1].keys(): 
            # Check for synonyms for node (if they exist)
                
            for synonym in node[1]["synonym"]:
                synonym_name = synonym.split("\"")[1]
                names.append(synonym_name)
                
    return names


def load_ctd_chemicals():
    """
    Loads CTD-chemicals vocabulary from respective local tsv file.

    :return: kb_graph, name_to_id, synonym_to_id, id_to_name 
    :rtype: Networkx MultiDiGraph object, dict, dict, id_to_name

    """
    
    id_to_name = dict()
    names = []

    with open("../../data/kb_files/CTD_chemicals.tsv") as ctd_chem:
        reader = csv.reader(ctd_chem, delimiter="\t")
        row_count = 0
        
        for row in reader:
            row_count += 1
            
            if row_count >= 30:
                chemical_name = row[0] 
                names.append(chemical_name)
                synonyms = row[7].split('|')

                for synonym in synonyms:
                    names.append(synonym)
    
    return names