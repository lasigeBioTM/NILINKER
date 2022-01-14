import argparse
import json
import logging
import sys
from utils import get_stats
sys.path.append("./")


def process_results(dataset, link, nil_linking):
    """Processes the results after the application of the PPR-IC model."""

    correct_count = int()
    wrong_count = int()
    total_count = int()
    nil_count = int()
    results_filepath = "results/REEL/" + dataset + "/" + link + "/" \
        + nil_linking + "_all_all"

    # Import PPR output
    with open(results_filepath, 'r') as results:
        data = results.readlines()
        results.close

    # Get outputted answer for each entity in each file 
    # and check if it's correct
    doc_id = str()
    doc_tmp = list()
    nil_count = int()

    for line in data:
        
        if line != "\n":
            
            if line[0] == "=":
            
                doc_id = line.strip("\n").split(" ")[1]
             
                if doc_id not in doc_tmp:
                    doc_tmp.append(doc_id)
                
            else:
                entity_text = line.split("\t")[1].split("=")[1] 
                number_of_mentions = int(line.split("\t")[0])
                total_count += 1 * number_of_mentions    
                correct_answer = line.split("\t")[2]                
                answer = line.split("\t")[3].strip("ANS=").strip("\n")

                if answer == correct_answer:
                    correct_count += number_of_mentions* 1
                        
                else:
                    
                    wrong_count += number_of_mentions* 1
                        
              
    doc_count = len(doc_tmp)

    return doc_count, correct_count, total_count#, nil_count
        
              
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
        total_unique_count = process_results(args.dataset, 
                                             args.link,
                                             args.nil_linking)

    # At this stage, we cannot acces the number of total entities initially 
    # present in corpus, so this is just a placeholder
    total_entities = 0

    get_stats(doc_count, total_entities, total_unique_count, 
                  correct_count, args)
  