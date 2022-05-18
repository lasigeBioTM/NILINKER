import argparse
import os

# Perform Named Entity Linking Evalution on several benchmarks using the 
# REEL-based models and BioSyn


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, required=True, 
        choices = ['bc5cdr_medic', 'bc5cdr_chem', 'gsc+', 'ncbi_disease', 'chr',
        'phaedra'])
    parser.add_argument("-model", type=str, required=True,
        choices=['biosyn', 'reel', 'reel_SM', 'reel_nilinker'] )
    parser.add_argument('-subset', type=str, required=True, 
        choices=['test', 'test_refined'])
    args = parser.parse_args()

    target_kbs = {'bc5cdr_medic': 'medic', 'bc5cdr_chem': 'ctd_chem', 
        'gsc+': 'hp', 'ncbi_disease': 'medic', 'chr': 'chebi', 
        'phaedra': 'chebi'}

    #--------------------------------------------------------------------------

    if args.model == 'biosyn':
        os.chdir('BioSyn-master/')

        if args.dataset == 'bc5cdr_medic':
            model_filepath = 'dmis-lab/biosyn-sapbert-bc5cdr-disease'
            dataset_up = 'bc5cdr-disease'

        elif args.dataset == 'bc5cdr_chem':
            model_filepath = 'dmis-lab/biosyn-sapbert-bc5cdr-chemical'
            dataset_up = 'bc5cdr-chemical'
        
        elif args.dataset == 'ncbi_disease':
            model_filepath = 'dmis-lab/biosyn-sapbert-ncbi-disease'
            dataset_up = 'ncbi-disease'
       
        comm = 'python inference.py -dataset {} -subset {} -model_name_or_path {} --dictionary_path datasets/{}/test_dictionary.txt --use_cuda --show_predictions'.\
            format(args.dataset, args.subset, model_filepath, dataset_up) 
        
        os.system(comm)

        os.chdir('..')

    #--------------------------------------------------------------------------
    elif args.model == 'reel' or args.model == 'reel_SM' \
            or args.model == 'reel_nilinker':

        links_dict = {'bc5cdr_medic': 'kb_corpus', 'bc5cdr_chem': 'kb_corpus', 'ncbi_disease': 'kb', 'chr': 'kb_corpus', 'gsc+': 'kb', 'phaedra': 'kb_corpus'}
        link_mode = links_dict[args.dataset]
        
        if args.model == 'reel':
            nil_linking = 'none'
        
        elif args.model == 'reel_SM':
            nil_linking = 'StringMatcher'
        
        elif args.model == 'reel_nilinker':
            nil_linking = 'NILINKER'
        
        #---------------------------------------------------------------------#
        #                       PRE_PROCESSING                                
        #        Pre-processes the corpus to create a candidates file for 
        #        each document in dataset to allow further building of the                 
        #        disambiguation graph.  
        #---------------------------------------------------------------------# 
        comm = 'python3 src/REEL/pre_process.py -dataset {} -subset {} -link {} -nil_linking {}'.\
            format(args.dataset, args.subset, link_mode, nil_linking)
        print(comm)
        os.system(comm)

        #---------------------------------------------------------------------#
        #                                     PPR                                     
        #         Builds a disambiguation graph from each candidates file:            
        #         the nodes are the candidates and relations are added                
        #         according to candidate link_mode. After the disambiguation          
        #         graph is built, it runs the PPR algorithm over the graph            
        #         and ranks each candidate.                                           
        #---------------------------------------------------------------------#
        comm = 'java -classpath :src/REEL/ ppr_for_ned_all {} ppr_ic {} {}'.\
            format(args.dataset, link_mode, nil_linking) 
        os.system(comm)

        #---------------------------------------------------------------------#
        #                               POST-PROCESSING                                
        #         Results file will be in dir results/REEL/$1/$2/                     
        #---------------------------------------------------------------------#
        comm = 'python3 src/REEL/post_process.py -dataset {} -link {} -nil_linking {}'.\
            format(args.dataset, link_mode, nil_linking) 
        os.system(comm)