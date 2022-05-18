#import json
import networkx as nx
#import os
from rapidfuzz import fuzz, process
from utils import candidate_string


def map_to_kb(entity_text, name_to_id, synonym_to_id, kb_cache):
    """
    Retrieves best KB matches for entity text according to lexical similarity 
    (edit distance).

    :param entity_text: the surface form of given entity 
    :type entity_text: str
    :param name_to_id: mappings between each KB concept name and 
        respective KB id
    :type name_to_id: dict
    :param synonym_to_id: mappings between each synonym for a 
        given KB concept and respective KB id
    :type synonym_to_id: dict
    :param kb_cache: mappings between entity mentions and KB candidates that 
        were previously found
    :type kb_cache: dict
    
    :return: matches (list) with format 
        [{kb_id: (..), name: (...), match_score: (...)}],
        and changed_cache (bool), which determines if cache will be filled 
        with a new candidates list for given entity  
    :rtype: tuple
    """
  
    changed_cache = False 
    top_concepts = list()

    if entity_text in name_to_id or entity_text in synonym_to_id: 
        # There is an exact match for this entity
        top_concepts = [entity_text]
    
    if entity_text.endswith("s") and entity_text[:-1] in kb_cache:
        # Removal of suffix -s 
        top_concepts = kb_cache[entity_text[:-1]]
    
    elif entity_text in kb_cache: 
        # There is already a candidate list stored in cache file
        top_concepts = kb_cache[entity_text]

    else:
        # Get first ten KB candidates according to lexical similarity 
        # with entity_text
        top_concepts = process.extract(
            entity_text, name_to_id.keys(), scorer=fuzz.token_sort_ratio, 
            limit=10)
        
        if top_concepts[0][1] == 100: 
            # There is an exact match for this entity
            top_concepts = [top_concepts[0]]
    
        elif top_concepts[0][1] < 100: 
            # Check for synonyms to this entity
            synonyms = process.extract(
                entity_text, synonym_to_id.keys(), limit=10, 
                scorer=fuzz.token_sort_ratio)

            for synonym in synonyms:

                if synonym[1] == 100:
                    top_concepts = [synonym]
                
                else:
                
                    if synonym[1] > top_concepts[0][1]:
                        top_concepts.append(synonym)
        
        kb_cache[entity_text] = top_concepts
        changed_cache = True
    
    # Build the candidates list with match id, name and matching score 
    # with entity_text
    matches = []
    
    for concept in top_concepts:
        
        term_name = concept[0]
        
        if term_name in name_to_id.keys():
            term_id = name_to_id[term_name]
        
        elif term_name in synonym_to_id.keys():
            term_id = synonym_to_id[term_name]
        
        else:
            term_id = "NIL"

        match = {"kb_id": term_id,
                 "name": term_name,
                 "match_score": concept[1]/100}
    
        matches.append(match)
  
    return matches, changed_cache, kb_cache


def update_entity_list(candidates_list, solution_number, 
                       solution_label_matches_entity):
    """
    Places the correct candidate in the first position of the candidates 
    list.

    :param candidates_list: includes retrieved KB candidates for an entity
    :type candidates_list: list
    :param solution_number: position of the correct KB candidate in the 
        candidates list
    :type solution_number: int
    :param solution_label_matches_entity: indicates if there is a 
        perfect match between entity string and correct KB candidate string
    :type solution_label_matches_entity: bool
    
    :return: updated_list, with correct KB candidate in the first position
    :rtype: list
    """

    updated_list = list()
    entity_perfect_matches = list()
    correct_candidate = candidates_list[solution_number]
    
    del candidates_list[solution_number]
    
    # There are perfect matches for this entity (some label lowercased)
    if solution_label_matches_entity:
        updated_list = [
            correct_candidate] \
            + [e for e in entity_perfect_matches[:] if e != correct_candidate]
    
    else:
        updated_list = [correct_candidate] + candidates_list
    
    return updated_list

    
def generate_candidates_list(entity_text, entity_id, kb, kb_graph, kb_cache, 
                             name_to_id, synonym_to_id, node_2_alt_ids, min_match_score,
                             nil_candidates=None):
    """
    Builds a structured candidates list for given entity.

    :param entity_text: string of the considered entity
    :type entity_text: str
    :param entity_id: the KB id associated with the concept to which entity
        should be linked (i.e. correct disambiguation)
    :type entity_id: str
    :param kb: the target KB
    :type kb: str
    :param kb_graph: Networkx object representing the ontology
    :type kb_graph: Networkx object 
    :param name_to_id: mappings between each ontology concept name and 
        the respective id
    :type name_to_id: dict
    :param synonym_to_id: mappings between each synonym for a given ontology 
        concept and the respective id
    :type synonym_to_id: dict
    :param node_2_alt_ids: mapings between each concept ID and its respective
        alternative IDs
    :type node_2_alt_ids: dict
    :param min_match_score: minimum edit distance between the mention text 
        and candidate string, candidates below this threshold are excluded 
        from candidates list
    :type min_match_score: float
    :param nil_candidates: the list of candidates retrieved for given NIL 
        entity (optional)
    :type nil_candidates: dict
    
    :return: candidates_list (list), solution_found (bool)
    :rtype: tuple
    """

    solution_found = False
    candidates_list  = list()
    less_than_min_score = int()
    
    # Retrieve best KB candidates names and respective ids
    candidate_names = list()

    if nil_candidates == None:
        candidate_names, changed_cache, kb_cache_up = map_to_kb(
            entity_text, name_to_id, synonym_to_id, kb_cache)
  
    else:
        candidate_names = nil_candidates

    # Get properties for each retrieved candidate 
    solution_number = -1
    nil_count = 0
    solution_label_matches_entity = False
    
    for i, candidate in enumerate(candidate_names): 
        
        if candidate["match_score"] > min_match_score \
                and candidate["kb_id"] != "NIL":
            
            outcount = kb_graph.out_degree(candidate["kb_id"])
            incount = kb_graph.in_degree(candidate["kb_id"])
            candidate_id = str()
            
            if kb == "medic" or kb == "ctd_chemicals":
                candidate_id = candidate["kb_id"]

                if '|' in candidate_id:
                    candidate_id = candidate_id.split('|')[0] 
                
                if candidate_id[:4] == "MESH":

                    if candidate_id == 'MESH_C' or candidate_id == 'MESH_D':
                        candidate_id = 10000000
                    
                    else:
                        candidate_id = int(candidate_id[6:])

                elif candidate_id[:4] == "OMIM":
                    candidate_id = int(candidate_id[6:])       
               
            elif kb == "chebi" or kb == "hp":
                candidate_id = int(candidate["kb_id"].split("_")[1])
            
            # The first candidate in candidate_names 
            # should be the correct disambiguation for entity
            candidates_list.append(
                {"url": candidate["kb_id"], "name": candidate["name"],
                "outcount": outcount, "incount": incount, "id": candidate_id, 
                "links": [], "score": candidate["match_score"]})

            match_found = False

            if entity_id == candidate["kb_id"] or (entity_id != '' and entity_id in candidate["kb_id"]):
                match_found = True
                  
            else:
                # FIND ALT IDS
                if candidate["kb_id"] in node_2_alt_ids.keys():
                    alt_ids = node_2_alt_ids[candidate["kb_id"]]

                    for alt_id in alt_ids:

                        if alt_id == entity_id:
                            match_found = True
                        
                            # The alt if will be the new entity id from now on
                            entity_id = alt_id

            if match_found:
                solution_number = i - nil_count - less_than_min_score

                if entity_text == candidate["name"]:
                    solution_label_matches_entity = True

        else:
            less_than_min_score += 1
   
    if solution_number > -1:
        # Update candidates list to put the correct answer as first 
        candidates_list = update_entity_list(
            candidates_list, solution_number, solution_label_matches_entity)
        solution_found = True
                
    else:
        
        if nil_candidates != None:
            return candidates_list
        
        else:
            candidates_list = []

    return candidates_list, entity_id, solution_found, changed_cache, kb_cache_up


def check_if_related(c1, c2, link_mode, extracted_relations, kb_edges):
    """
    Checks if two given KB concepts/candidates are linked according to the 
    criterium defined by link_mode.

    :param c1: KB concept/candidate 1
    :type c1: str
    :param c2: KB concept/candidate 2
    :type c2: str
    :param link_mode: how the edges are added to the disambiguation graph ('kb',
    'corpus', 'kb_corpus')
    :type link_mode: str
    :param extracted_relations: relations extracted from target corpus
    :type extracted_relations: list
    :param kb_edges: relations described in the knowledge base
    :type kb_edges: list
    
    :return: related, is True if the two candidates are related, False 
             otherwise
    :rtype: bool

    :Example:

    >>> c1 = "ID:01"
    >>> c2 = "ID:02"
    >>> link_mode = "corpus"
    >>> extracted_relations = {"ID:01": ["ID:02"], "ID:03": ["ID:02"]} 
    >>> kb_edges = ["ID:04_ID:O5", "ID:06_ID:07"]
    >>> check_if_related(c1, c2, link_mode, extracted_relations, kb_edges)
    True

    """
    
    rel_str1 = c1 + "_" + c2[0]
    rel_str2 = c2[0] + "_" + c1

    related = False

    if link_mode == "corpus":
        # Check if there is an extracted relation between the
        # two candidates
        if c1 in extracted_relations.keys():
            relations_with_c1 = extracted_relations[c1]
                            
            if c2[0] in relations_with_c1: 
                # Found an extracted relation
                related = True

    else:

        if c1 == c2[0] \
                or rel_str1 in kb_edges \
                or rel_str2 in kb_edges: 
            # There is a KB link between the two candidates
            related = True
                        
        else:
            # There is no KB link between the two candidates
                                                       
            if link_mode == "kb_corpus": 
                # Maybe there is an extracted relation 
                # between the two candidates
                                
                if c1 in extracted_relations.keys():
                    relations_with_c1 = extracted_relations[c1]
                            
                    if c2[0] in relations_with_c1: 
                        # Found an extracted relation
                        related = True
    
    return related


def write_candidates_file(doc_entities_candidates, candidates_filename, 
                          entity_type, kb_graph, link_mode, 
                          extracted_relations):
    """
    Generates the candidates file associated with given document in the 
    corpus. 

    :param doc_entities_candidates: includes entities and respective 
        candidates to output
    :type doc_entities_candidates: dict
    :param candidates_filename: filename for output candidates file
    :type candidates_filename: str
    :param entity_type: "Chemical", "Disease"
    :type entity_type: str
    :param kb_graph: represents the target knowledge base
    :type kb_graph: Networkx MultiDiGraph object
    :param link_mode: how to add edges in the disambigution graphs: kb, corpus,
        or corpus_kb"
    :type link_mode: str
    :param extracted_relations: includes extracted relations or is empty if
        link_mode=kb
    :type extracted_relations: list
    
    :return: entities_written, representing the number of entities with at 
        least one candidate and that were included in the candidates file
    :rtype: int
    
    """
    
    entities_written = int() # Entitites with at least one candidate
    candidates_links = dict() 

    candidates_file = open(candidates_filename, 'w')
    
    for annotation in doc_entities_candidates:
        entities_written += 1
        entity_str = annotation[0]
        candidates_file.write(entity_str)
        
        # Iterate on candidates for current entity
        for c in annotation[1]:
            
            if c["url"] in candidates_links:
                c["links"] = candidates_links[c["url"]]
    
            else: 
                links = list()
                other_candidates = list()

                # Iterate over candidates for every other entity except the 
                # current one to find links between candidates for different
                # entities (candidates for the same entity cannot be linked) 
                for annotation_2 in doc_entities_candidates:   
                    
                    if annotation_2 != annotation:
                        
                        for c2 in annotation_2[1]:
                            other_candidates.append((c2["url"], c2["id"]))
    
                for c2 in other_candidates: 
                    c1 = c["url"]
                    related = check_if_related(
                        c1, c2, link_mode, 
                        extracted_relations, kb_graph.edges())
                    
                    if related:
                        links.append(str(c2[1]))

                c["links"] = ";".join(set(links))
                candidates_links[c["url"]] = c["links"][:]

            candidates_file.write(
                candidate_string.format(c["id"], c["incount"], c["outcount"], 
                c["links"], c["url"], c["name"], c["name"].lower(), 
                c["name"].lower(), entity_type))
    
    candidates_file.close()
    
    return entities_written