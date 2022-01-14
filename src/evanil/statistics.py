import json
import os
import sys
from src.utils.utils import retrieve_annotations_from_evanil


def get_corpus_statistics(partition, annotations):
    """Calculates and outputs the corpus statistics.

    :param annotations: has format 
        {file_id: {annotation_str: [KB_id, direct_ancestor]}}
    :type annotations: dict
    """
    
    total_docs = len(annotations.keys())
    annot_count = int()
    word_1 = int()
    word_2 = int()
    word_3 = int()
    word_4 = int()
    word_5_more = int()
    
    for doc in annotations.keys():
        doc_annotations = annotations[doc]
        
        for annotation in doc_annotations.keys():  
            annot_count += 1              
            annotations_words = annotation.split(" ")
            
            if len(annotations_words) == 1:
                word_1 += 1
                    
            elif len(annotations_words) == 2:
                word_2 += 1
                    
            elif len(annotations_words) == 3:
                word_3 += 1

            elif len(annotations_words) == 4:
                word_4 += 1

            elif len(annotations_words) >= 5:
                word_5_more += 1
        
    stats = "Total number of annotations: " + str(annot_count) \
        + "\nTotal docs:" + str(total_docs) + "\n" \
        + str(word_1) + " annotations (" + str(float(word_1/annot_count)*100) \
            + " %) with 1 word\n" \
        + str(word_2) + " annotations (" + str(float(word_2/annot_count)*100) \
            + " %) with 2 words\n" \
        + str(word_3) + " annotations (" + str(float(word_3/annot_count)*100) \
            + " %) with 3 words\n" \
        + str(word_4) + " annotations (" + str(float(word_4/annot_count)*100) \
            + " %) with 4 words\n" \
        + str(word_5_more) + " annotations (" \
            + str(float(word_5_more/annot_count)*100) \
            + " %) with 5 or more words"

    out_dir = 'data/evanil/'
    
    with open(out_dir + partition + '_stats', 'w') as out_file:
        out_file.write(stats)
        out_file.close()

    print(stats)


if __name__ == "__main__":
    partition = sys.argv[1]
    
    all_annotations = retrieve_annotations_from_evanil(partition)
    get_corpus_statistics(partition, all_annotations)
    