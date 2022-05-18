import json
import annotations as annot
import os
import sys
sys.path.append(".")
from kbs import load_ctd_chemicals, load_obo

def refine_test_set(train_annots, test_annots, names):
    #TODO if appears on train and dev set change
    out_test_refined = {}
    train_texts = [annot[1].lower() for doc in train_annots.keys() for annot in train_annots[doc]]
    names_up = [name.lower() for name in names]
    train_texts.extend(names_up)
    refined_test = 0
    test = 0
  
    for doc_id in test_annots:

        for annot in test_annots[doc_id]:
            test +=1
            
            if annot[1].lower() not in train_texts:
                refined_test += 1
                #print(annot[1].lower())
                if doc_id in out_test_refined.keys():
                    #doc_annots_text = [annot for annot in out_test_refined[doc_id].keys()
                    out_test_refined[doc_id].append(annot)
                    
                else:
                    out_test_refined[doc_id] = [annot]
                    #
    print('refined mentions', str(refined_test))
    print('test mentions', str(test))
    return out_test_refined


if __name__ == '__main__':
    """Prepares NEL dataset by generating JSON files containing the annotations
    of the respective ('test set') and of the refined test set ('test_refined')"""

    dataset = sys.argv[1]

    #data_dirs = {}
    #dataset_dir = data_dirs[dataset]
    out_test = {}

    #{'doc_id': [annotation1, annotation2]}
    #annotation = (kb_id, annotation_text)

    if dataset == 'chr':
        names = load_obo('chebi')
        print(names)
        train_annots, test_annots = annot.parse_Pubtator('chr', 'CHR_corpus/')#,kb_ids) 
    
    elif dataset == 'bc5cdr_medic' or dataset == 'ncbi_disease':
        entity_type = 'Disease' 

        if dataset == 'bc5cdr_medic':
            dataset_dir = 'BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/'
            data_info = 'bc5cdr'
            

        elif dataset == 'ncbi_disease':
            dataset_dir = 'NCBI_disease_corpus/'
            data_info = 'ncbi_disease'
            
        names  = load_obo('medic')
        
        train_annots, test_annots = annot.parse_Pubtator(
                data_info, dataset_dir, entity_type=entity_type)
       
    elif dataset == 'bc5cdr_chem' or dataset == 'phaedra':
        entity_type = 'Chemical'
        names  = load_ctd_chemicals()
        
        if dataset == 'bc5cdr_chem':
            dataset_dir = 'BioCreative-V-CDR-Corpus/CDR_Data/CDR.Corpus.v010516/' 
            train_annots, test_annots = annot.parse_Pubtator(
                'bc5cdr', dataset_dir, entity_type=entity_type)

        elif dataset == 'phaedra':
            train_annots, test_annots  = annot.parse_phaedra_corpus()
        
    elif dataset == 'gsc+':
        names  = load_obo('hp')
        test_annots = annot.parse_GSC_corpus()
    
    #print(len(test_annots.keys()))
    #-------------------------------------------------------------------------
    #                           OUTPUT FILES
    #-------------------------------------------------------------------------
    out_dir = '../../data/corpora/preprocessed/{}/'.format(dataset)
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_test_json = json.dumps(test_annots, indent=4)
    out_filepath1 = out_dir + 'test.json'
    with open(out_filepath1, 'w') as out_file1:     
        out_file1.write(out_test_json)
        out_file1.close()

    if dataset != 'gsc+':
        test_annots_refined = refine_test_set(train_annots, test_annots, names)
        out_test_refined_json = json.dumps(test_annots_refined, indent=4)
        out_filepath2 = out_dir + 'test_refined.json'
        with open(out_filepath2, 'w') as out_file2:     
            out_file2.write(out_test_refined_json)
            out_file2.close()
    
    #-------------------------------------------------------------------------
    #                           Print statistics
    #-------------------------------------------------------------------------