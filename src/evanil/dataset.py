import argparse
import json
import logging
import sys
from annotations import parse_annotations

sys.path.append("./")


def build_partition(partition):
    """Generates the <partition>.json file in the dir 'data/evanil/ containing
    the partition annotations.

    :param annotations: has format 
        {file_id: {annotation_str: [KB_id, direct_ancestor]}}
    :type annotations: dict
    :param partition: has value 'medic', 'ctd_chem', 'chebi', 'hp', 'ctd_anat'
    :type partition: str
    """

    logging.info('Parsing annotations...')

    has_pbmds_files = ['medic', 'ctd_anat', 'ctd_chem']
    annotations = dict()

    if partition in has_pbmds_files:   
        # PBDMS dataset is too large, so each split
        # is processed sequentially.
        
        #######################################################################
        # To generate only the splits used in the experiments
        
        # For 'medic' or ctd_'anat':
        # splits = ['0', '1', '2', '3']
        
        # For ctd_chem (only split 0):
        #splits = ['0']

        #######################################################################
        # To generate the entire EvaNIL dataset (0 to 29 splits):
        splits = list(range(0, 30))
       
        for split in splits:
            split_annotations = parse_annotations(
                                    partition, split=split)
            annotations.update(split_annotations)
        
    else:
        annotations = parse_annotations(partition)

    logging.info('Generating annotations file...')

    out_dir = "./data/evanil/"         
    out_dict = json.dumps(annotations, indent=4, ensure_ascii=False)
    out_filename = out_dir + partition + ".json"

    with open(out_filename, "w", encoding="utf-8") as out_file:
        out_file.write(out_dict)
        out_file.close()
    
    logging.info('Done!')
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                prog="dataset", 
                description="Builds the selected partition of the EvaNIL \
                            dataset")
    parser.add_argument(
        "-partition", 
        type=str, 
        choices=['hp', 'chebi', 'medic', 'ctd_chem', 'ctd_anat', 'go_bp'],
        help="The seleccted partition to build.")               
    args = parser.parse_args()
    
    log_filename = 'logs/evanil/' + args.partition + '.log'
    logging.basicConfig(
            filename=log_filename, level=logging.INFO, 
            format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
            filemode='w')
    
    build_partition(args.partition)