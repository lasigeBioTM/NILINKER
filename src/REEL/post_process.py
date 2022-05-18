import argparse
#import json
#import logging
import sys
from utils import get_stats
sys.path.append("./")


def process_results(dataset, link, nil_linking):
    """Processes the results after the application of the PPR-IC model."""

    correct_count = int()
    wrong_count = int()
    total_count = int()
    #nil_count = int()
    results_filepath = "results/REEL/" + dataset + "/" + link + "/" \
        + nil_linking + "_all_all"

    # Import PPR output
    with open(results_filepath, 'r') as results:
        data = results.readlines()
        results.close

    # Get outputted answer for each entity in each file 
    # and check if it's correct
    doc_id = ''
    doc_tmp = []
    nil_count = 0

    for line in data:
        
        if line != "\n":
            
            if line[0] == "=":
                doc_id = line.strip("\n").split(" ")[1]
             
                if doc_id not in doc_tmp:
                    doc_tmp.append(doc_id)
                
            else:
                number_of_mentions = int(line.split("\t")[0])
                total_count += number_of_mentions    
                correct_answer = line.split("\t")[2]                
                answer = line.split("\t")[3].strip("ANS=").strip("\n")
              
                if correct_answer == 'MESH_-1':
                    nil_count += number_of_mentions
                
                elif correct_answer == '-1':
                    nil_count += number_of_mentions
            
                else:
                    
                    if answer == correct_answer or answer in correct_answer:                 
                        correct_count += number_of_mentions * 1
                        
                    else:
                        wrong_count += number_of_mentions * 1
                          
    doc_count = len(doc_tmp)

    return doc_count, correct_count, wrong_count, total_count, nil_count
        
              
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-dataset", type=str, required=True,
        help="bc5cdr_medic_test, GSC+")
    parser.add_argument("-link", type=str, required=True,
        help="How to add edges in the disambigution graphs: kb, corpus, \
              corpus_kb")
    parser.add_argument("-nil_linking", type=str, required=False,
        help="none, StringMatcher or NILINKER. Defaults to None")
    
    args = parser.parse_args()

    doc_count, correct_count, \
        wrong_count, total_count, nil_count = process_results(args.dataset, 
                                             args.link,
                                             args.nil_linking)

    get_stats(doc_count, total_count, nil_count,
                  correct_count, wrong_count, args, final=True)
  