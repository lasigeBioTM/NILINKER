import argparse
import sys
import json
from src.REEL.utils import stringMatcher
from src.NILINKER.predict_nilinker import load_model
from src.utils.kbs import KnowledgeBase
from tqdm import tqdm
#from ..BioSyn.BioSyn_master.inference import main


def calculate_metrics(tp,fp,fn):

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = tp / (tp+fp)

    return accuracy, precision, recall, f1_score 
    #precision, recall, f1_score, accuracy


def filter_top_pred(predictions, true_kb_id):
    """If the top prediction is associated with the true_kb_id, the top 
    prediction is replaced by the first prediction that is not associated with 
    the true_kb_id"""

    prediction_found = False
    top_prediction = None

    for pred in predictions:

        if pred[0] != true_kb_id:
            prediction_found = True
            top_prediction = pred[0]
        
        else:
            "This is wrong"

        if prediction_found:
            break

    return top_prediction
    

def import_input(filepath):
    """Import mentions from json file into dict"""

    gold_standard = {}

    with open(filepath, 'r') as in_file:
        gold_standard = json.load(in_file)
        in_file.close()

    return gold_standard


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-partition", type=str, required=True)
    parser.add_argument("-model", type=str, required=True)
    parser.add_argument("--refined", type=bool, default=False)
    args = parser.parse_args()

    # Load selected model
    loaded_model = None

    if args.model == 'nilinker':
        nilinker = load_model(args.partition, top_k=10)
    
    elif args.model == 'stringMatcher':
        obo_list = ['hp', 'chebi', 'medic', 'go_bp'] 
        tsv_list = ['ctd_chem', 'ctd_anat']
        
        kb_data = KnowledgeBase(args.partition)

        if args.partition in obo_list:
            kb_data.load_obo(kb_data.kb)  

        elif args.partition in tsv_list:    
            kb_data.load_tsv(kb_data.kb) 

        named_to_id = kb_data.name_to_id

    elif args.model == 'biosyn':
        top_k=10
        biosyn = None

    # Import mentions of the test set
    test_filepath = ''

    if args.refined:
        test_filepath = 'data/evanil/{}/test_refined.json'.format(args.partition)

    else:
        test_filepath = 'data/evanil/{}/test.json'.format(args.partition)
        print('test')
    gold_standard = import_input(test_filepath)

    # Iterate over each mention and apply model
    tp = 0
    fp = 0
    fn = 0

    pbar = tqdm(total=len(gold_standard.keys()))

    for doc in gold_standard:
        mentions = gold_standard[doc]

        for mention in mentions.keys(): 
            ancestor_id = mentions[mention][1]
            true_kb_id = mentions[mention][0]
            
            if args.model == 'nilinker':    
                predictions = nilinker.prediction(mention)
                top_pred_id = filter_top_pred(predictions, true_kb_id)
                #print("ANCESTOR ID", ancestor_id, "\tTOP PRED", top_pred_id)
            elif args.model == 'biosyn':
                predictions = (mention)
                top_pred_id = filter_top_pred(predictions, true_kb_id)

            elif args.model == 'stringMatcher':
                predictions = stringMatcher(mention, named_to_id, 10)
                top_pred_id = filter_top_pred(predictions, true_kb_id)
        
            if top_pred_id == ancestor_id:
                tp += 1
          
            elif top_pred_id != ancestor_id:

                if top_pred_id == None or top_pred_id == '':
                    fn += 1
                
                else:
                    fp += 1 

        pbar.update(1)

    pbar.close()
    print("tp:", tp, "\nfp:", fp, "\nfn:", fn)
    accuracy, precision, recall, f1_score = calculate_metrics(tp,fp,fn)
    #precision, recall, f1_score, accuracy
    print("Results\nPrecision: {}\nRecall: {}\nF1-score: {}\nAccuracy: {}".format(
        precision, recall, f1_score, accuracy))