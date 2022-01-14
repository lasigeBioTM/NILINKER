import annotations as annot
import argparse
import json
import logging
import os
import sys
from candidates import write_candidates_file, generate_candidates_list
from information_content import generate_ic_file
from kbs import load_obo, load_ctd_chemicals
from src.NILINKER.predict_nilinker import load_model
#from src.utils.kbs import KnowledgeBase
from relations import import_bolstm_output, import_cdr_relations_pubtator, import_chr_relations, import_hp_relations, import_phaedra_relations
from utils import entity_string, get_stats, stringMatcher
from tqdm import tqdm

sys.path.append("./")
        

def build_entity_candidate_dict(dataset, kb, entity_type, annotations, 
                                min_match_score, kb_graph, kb_cache, name_to_id, 
                                synonym_to_id, top_k, nil_model_name='none', 
                                nilinker=None):
    """
    Builds a dict including the candidates for all entity mentions in all corpus
    documents.

    :param dataset: the target dataset from where annotations were parsed
    :type dataset: str 
    :param kb: "medic", "chebi" or "ctd_chemicals"
    :type kb: str
    :param annotations: with format {doc_id:[(annot1_id, annot1_text)]}
    :type annotations: dict
    :param min_match_score: minimum edit distance between the mention text and 
        candidate string, candidates below this threshold are excluded from 
        candidates list
    :type min_match_score: float
    :param kb_graph: Networkx object representing the kb
    :type kb_graph: Networkx object
    :param name_to_id: mappings between each kb concept name and 
        the respective id
    :type name_to_id: dict
    :param synonym_to_id: mappings between each synonym for a given kb concept 
        and the respective id
    :type synonym_to_id: dict
    :param nil_linking_model: approach to disambiguate NIL entities, it must 
        be 'none', 'StringMatcher', or 'NILINKER'
    :type nil_linking_model: str
    
    :return: entities_candidates (dict) with format 
        {doc_id: {mention:[candidate1, ...]} }, changed_cache_final (bool) 
        indicating wether the candidates cache has been updated comparing with
        preivous execution of the script, and kb_cache_up (dict), which 
        corresponds to the candidates cache for given KB, updated or not 
        according to the value of changed_cache
    :rtype: tuple with dict, bool, dict
    
    """

    doc_count = int()
    total_entities_count = int()
    unique_entities_count = int()
    solution_is_first_count = int()
    nil_count = int()
    entities_candidates = dict() 
    nil_entities = list()
    changed_cache_final = False
    
    logging.info('Building entities-candidates dictionaries...')
    print('Building entities-candidates dictionaries...')
    
    doc_total = len(annotations.keys())
    pbar = tqdm(total=doc_total)
    
    for document in annotations.keys(): 
        doc_count += 1 
        document_entities = list()
        doc_entities = list()
        
        for annotation in annotations[document]:
            
            annotation_id = annotation[0]
            entity_text = annotation[1]
            normalized_text = entity_text
            total_entities_count += 1
            
            if normalized_text not in document_entities:
                # Repeated instances of the same entity in a document 
                # are not considered
                document_entities.append(normalized_text)
                unique_entities_count += 1
                
                # Get candidates list for entity
                candidates_list, \
                    solution_is_first, \
                    changed_cache, kb_cache_up = generate_candidates_list(
                                                    normalized_text, 
                                                    annotation_id, kb, 
                                                    kb_graph, 
                                                    kb_cache, name_to_id, 
                                                    synonym_to_id, 
                                                    min_match_score)
                
                if changed_cache:
                    #There is at least 1 change in the cache file
                    changed_cache_final = True

                # Check the solution found for this entity
                if solution_is_first: 
                    # The solution found is the correct one
                    solution_is_first_count += 1
                
                if len(candidates_list) == 0 or \
                        annotation_id == None or \
                        annotation_id == '' or \
                        annotation_id == '-1': 
                    
                    nil_count += 1 
                    
                    if nil_model_name != 'none':
                        # We will try to disambiguate the NIL entity
                        top_candidates_up = list()

                        if nil_model_name == 'NILINKER':
                            # Find top-10 candidates with NILINKER and include
                            # in the candidates file
                            top_candidates = nilinker.prediction(
                                                normalized_text)
                                            
                        elif nil_model_name == 'StringMatcher':
                            top_candidates = stringMatcher(
                                                    normalized_text, 
                                                    name_to_id,
                                                    1)
                            
                        for cand in top_candidates:
                            kb_id = cand[0]

                            if nil_model_name == 'NILINKER':
                                kb_id = cand[0].replace(":", "_")
                        
                            cand_up = {'kb_id': kb_id , 
                                        'name': cand[1],
                                        'match_score': 1.0}
                            
                            top_candidates_up.append(cand_up)
                        
                        candidates_list = generate_candidates_list(
                            entity_text, '', kb, 
                            kb_graph, kb_cache, 
                            name_to_id, synonym_to_id, 
                            min_match_score,
                            nil_candidates=top_candidates_up)
                        
                    elif nil_model_name == 'none':
                        # Since nil entities are not disambiguated,
                        # create a dummy candidate
                        candidates_list = [
                            {'url': 'NIL', 
                            'name': 'none', 
                            'outcount': 0, 
                            'incount': 0, 
                            'id': -1, 'links': [], 
                            'score': 0}]
                    
                entity_type = entity_type                
                entity_str = entity_string.format(
                    entity_text, normalized_text, entity_type, 
                    doc_count, document, annotation_id)
                
                add_entity = [entity_str, candidates_list]
                doc_entities.append(add_entity)

        entities_candidates[document] = doc_entities
            
        pbar.update(1)
        
    pbar.close()
    doc_count = len(entities_candidates.keys())
    
    get_stats(
        doc_count, total_entities_count, unique_entities_count,
        solution_is_first_count, args)

    return entities_candidates, changed_cache_final, kb_cache_up
    

def pre_process(args):
    """
    Executes all necessary pre-processing steps necessary to create the 
    candidate files, which are the input for the PPR algorithm.

    :param args: includes the arguments defined by the user, such as 'dataset',
        'link' and 'nil_linking'
    :type args: ArgumentParser object
    """

    logging.info('Pre-processing')

    # Min lexical similarity between entity text and candidate text: 
    # exclude candidates with a lexical similarity below min_match_score
    min_match_score = 0.0 
    
    kb_graph = None
    name_to_id = dict()
    synonym_to_id =  dict()
    annotations =  dict()    
    kb_name = str()
    entity_type = str()
    top_k = int() # Top candidates that NILINKER returns
    kb_cache = dict()

    if args.dataset == 'chr' or args.dataset == 'chebiPatents':
        # TARGET KB: CHEBI
        kb_graph, name_to_id, synonym_to_id, id_to_name  = load_obo('chebi')  
        kb_name = 'chebi'
        entity_type = 'Chemical'
        
        if args.dataset == 'chr':
            dataset_dir = 'CHR_corpus/'
            annotations = annot.parse_Pubtator('chr', dataset_dir ) 
        
        elif args.dataset == 'chebiPatents':
            annotations = annot.parse_chebi_patents()
            top_k = 5
        
    elif args.dataset == 'bc5cdr_medic' or args.dataset == 'ncbi_disease':
        # TARGET KB: MEDIC
        kb_graph, name_to_id, synonym_to_id, \
            id_to_name  = load_obo('medic')
        
        kb_name = 'medic'
        entity_type = 'Disease' 

        if args.dataset == 'bc5cdr_medic':
            dataset_dir = 'BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/'
            data_info = 'bc5cdr'
            top_k = 2 # optimized in 100 docs

        elif args.dataset == 'ncbi_disease':
            dataset_dir = 'NCBI_disease_corpus/'
            data_info = 'ncbi_disease'
            top_k = 2

        annotations = annot.parse_Pubtator(
                data_info, dataset_dir, entity_type=entity_type)
            
    elif args.dataset == 'bc5cdr_chem' or args.dataset == 'phaedra':
        # TARGET KB: CTD-CHEM
        kb_graph, name_to_id, synonym_to_id, \
            id_to_name  = load_ctd_chemicals()
        
        kb_name = 'ctd_chemicals'
        entity_type = 'Chemical'
        
        if args.dataset == 'bc5cdr_chem':
            dataset_dir = 'BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/'
            
            annotations = annot.parse_Pubtator(
                'bc5cdr', dataset_dir, entity_type=entity_type)
            
            top_k = 5  
            
        elif args.dataset == 'phaedra':
            annotations = annot.parse_phaedra_corpus()
            
            top_k = 5

    elif args.dataset == 'gsc+':
        kb_graph, name_to_id, synonym_to_id, id_to_name  = load_obo('hp')
        kb_name = 'hp'
        entity_type = 'Disease'
        annotations = annot.parse_GSC_corpus()
        top_k = 6
        
    kb_cache_filename = 'data/REEL/cache/{}.json'.format(kb_name)
    kb_cache = dict()

    if os.path.exists(kb_cache_filename):
        cache_file = open(kb_cache_filename)
        kb_cache = json.load(cache_file)
        cache_file.close()

    changed_cache_final = False

    if args.nil_linking == 'NILINKER':
        # Prepare NILINKER and get compiled model ready to predict
        # top_k defines the number of candidates that NILINKER wil return
        # for each given entity
        kb_name_up = str()

        if kb_name == 'ctd_chemicals':
            kb_name_up = 'ctd_chem'
        
        else:
            kb_name_up = kb_name

        print('Loading NILINKER...')
        nilinker = load_model(kb_name_up, top_k=int(top_k))
      
    else:
        nilinker = None

    # Check if results dir exists
    results_dir = 'results/REEL/' + args.dataset + '/' 

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    entities_candidates, \
        changed_cache_final, kb_cache_up = build_entity_candidate_dict(
                                            args.dataset, kb_name, entity_type,
                                            annotations, 
                                            min_match_score, kb_graph, 
                                            kb_cache, name_to_id, 
                                            synonym_to_id, 
                                            top_k,
                                            nil_model_name=args.nil_linking,
                                            nilinker=nilinker)
    
    if changed_cache_final:
        logging.info('Updating KB cache file')
        cache_out = json.dumps(kb_cache_up)
    
        with open(kb_cache_filename, 'w') as cache_out_file:
            cache_out_file.write(cache_out)
            cache_out_file.close()

    logging.info('Getting extracted relations')
    
    extracted_relations = dict()

    if args.link == 'corpus' or args.link == 'kb_corpus': 
            
        if args.dataset == 'craft_chebi':
            extracted_relations = import_bolstm_output()
            
        elif args.dataset == 'bc5cdr_medic' or \
                args.dataset == 'bc5cdr_chem':
            extracted_relations = import_cdr_relations_pubtator(
                args.dataset, "all")
            
        elif args.dataset == 'chr':
            extracted_relations = import_chr_relations()
        
        elif args.dataset == 'gsc+':
            extracted_relations = import_hp_relations()
        
        elif args.dataset == 'phaedra':
            extracted_relations = import_phaedra_relations()
        
        else:
            print('There are no extracted relations available for this dataset!')

    logging.info('Generating candidates files')
    print('Generating candidates files...')
    
    candidates_dir = 'data/REEL/candidates/' + args.dataset + '/' 

    if not os.path.exists(candidates_dir):
        os.mkdir(candidates_dir)

    candidates_dir_2 = candidates_dir + args.link + '/'
    
    if not os.path.exists(candidates_dir_2):
        os.mkdir(candidates_dir_2)
    
    candidates_dir_3 = candidates_dir_2  + args.nil_linking + '/'

    if not os.path.exists(candidates_dir_3):
        os.mkdir(candidates_dir_3)
    
    # Delete existing candidates files
    cand_files = os.listdir(candidates_dir_3)

    if len(cand_files)!=0:
        
        for file in cand_files:
            os.remove(candidates_dir_3 + file)

    doc_count = int() 
    entities_writen = int()
    pbar = tqdm(total=len(entities_candidates.keys()))
        
    for document in entities_candidates:
        doc_count += 1
        candidates_filename = candidates_dir_3 + document
        entities_writen += write_candidates_file(
            entities_candidates[document], candidates_filename, 
            entity_type, kb_graph, args.link, extracted_relations)
        pbar.update(1)

    pbar.close()
    
    logging.info('Done! {} entities written in {} candidates files:'.\
        format(str(entities_writen), str(doc_count)))
        
    # Create information content file including every KB concept
    # appearing in candidates files 
    generate_ic_file(args.dataset, args.link, args.nil_linking, 
                         annotations)

    # Generate results dir 
    results_dir = './results/REEL/' + args.dataset + '/' + args.link + '/' 
    
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    logging.info('Pre-processing finished!')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-dataset", type=str, required=True,
        choices = ['bc5cdr_medic', 'bc5cdr_chem', 'gsc+', 'ncbi_disease',
                   'chr', 'phaedra', 'chebiPatents'],
        help= 'The target dataset containing the entities to link')
    parser.add_argument('-link', type=str, required=True,
        choices = ['kb', 'corpus', 'kb_corpus'], 
        help='How to add edges in the disambigution graphs: kb, corpus, \
              corpus_kb')
    parser.add_argument('-nil_linking', type=str, required=True,
        choices = ['none', 'StringMatcher', 'NILINKER'],
        help='Approach to deal with NIL entities')

    args = parser.parse_args()
    
    log_dir = 'logs/REEL/pre_process/' + args.dataset + '/'

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    log_filename = log_dir + args.link + '_' + \
        args.nil_linking + '.log'
    
    logging.basicConfig(
        filename=log_filename, level=logging.INFO, 
        format='%(asctime)s | %(levelname)s: %(message)s', 
            datefmt='%m/%d/%Y %I:%M:%S %p',
        filemode='w')

    entity_types = {'chebi': 'Chemical', 'ctd_chemicals': 'Chemical', 
                    'medic': 'Disease', 'hp': 'Disease'}
                    
    pre_process(args)