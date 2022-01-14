import os
import networkx as nx
import sys
import xml.etree.ElementTree as ET
from annotations import parse_cdr_annotations_pubtator, \
    parse_craft_chebi_annotations
from math import log
sys.path.append("./")


def build_extrinsic_information_content_dict(annotations):
    """Generates dictionary with extrinsic information content (Resnik's) 
       for each term appearing in corpus.""" 

    term_counts = dict()
    extrinsic_ic = dict()
   
    # Get the term frequency in the corpus
    for document in annotations: 
        
        for annotation in annotations[document]:
            
            term_id = annotation[0]
            
            if term_id not in term_counts.keys():
                term_counts[term_id] = 1
            else:
                term_counts[term_id] += 1

    # Frequency of the most frequent term in dataset    
    max_freq = max(term_counts.values()) 
    
    for term_id in term_counts:
        
        term_frequency = term_counts[term_id] 
   
        term_probability = (term_frequency + 1)/(max_freq + 1)
    
        information_content = -log(term_probability) + 1
        
        extrinsic_ic[term_id] = information_content + 1
    
    return extrinsic_ic


def generate_ic_file(dataset, link_mode, nil_linking, annotations):
    """Generates file with information content of all entities referred 
       in candidates file."""

    out_string = str()
    
    candidates_dir = "data/REEL/candidates/{}/{}/{}/".format(dataset, 
                                                             link_mode, 
                                                             nil_linking)
    
    ic_dict = build_extrinsic_information_content_dict(annotations) 
    
    url_temp = list()

    for file in os.listdir(candidates_dir): 
        data = ''
        path = candidates_dir + file
        candidate_file = open(path, 'r', errors="ignore")
        data = candidate_file.read()
        candidate_file.close()
        
        for line in data.split('\n'):
            url, surface_form = str(), str()
            
            if line[0:6] == "ENTITY":
                surface_form = line.split('\t')[1].split('text:')[1]
                url = line.split('\t')[8].split('url:')[1]
                
            
            elif line[0:9] == "CANDIDATE":
                surface_form = line.split('\t')[6].split('name:')[1]
                url = line.split('\t')[5].split('url:')[1]
          
            if url not in url_temp:
                    
                if url != "":   
                    url_temp.append(url)

                    if url in ic_dict.keys():
                        ic = ic_dict[url]                        
                                
                    else:
                        ic = 1.0
                        
                    if dataset == "craft_chebi":
                        url = url.split(":")[1]
                        out_string += "url:http:" + url +'\t' \
                            + str(ic) + '\n'
                        
                    else:
                        out_string += url +'\t' + str(ic) + '\n'
                        
    # Create file ontology_pop with information content for all entities 
    # in candidates file
    out_dir = "data/REEL/ic/" + dataset + "/"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    output_file_name = out_dir + link_mode + "_" + nil_linking + "_ic"
    
    with open(output_file_name, 'w') as ic_file:
        ic_file.write(out_string)
        ic_file.close()