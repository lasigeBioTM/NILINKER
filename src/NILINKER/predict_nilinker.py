import argparse
import logging
import os 
import tensorflow as tf
from src.NILINKER.nilinker import Nilinker
from src.utils.utils import get_wc_embeds, get_kb_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_model(partition, top_k=1):
    """Prepares the model for later prediction when requested.

    :param partition: has value 'medic', 'ctd_chem', 'hp', 'chebi', 
        "go_bp", "ctd_anatomy"
    :type partition: str
    
    :return: loaded and compiled NILINKER model for the specified partition 
    :rtype: tf.keras.Model() object 
    """

    word_embeds, candidate_embeds, wc, embeds_word2id = get_wc_embeds(partition)
    params = [200, wc.candidate_num, top_k]
    kb_data = get_kb_data(partition)
    
    model_dir = "./data/nilinker_files/{}/final/".format(partition)
    
    model = Nilinker(word_embeds, candidate_embeds, 
                     params, wc, kb_data, embeds_word2id)
    model.compile(run_eagerly = True)
    model.built = True
    model.load_weights(model_dir + "best.h5")
    print("-----> NILINKER ready!")

    return model


if __name__ == "__main__":

    # Info about available gpus
    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(3)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # Select all available gpus
    #os.environ["CUDA_VISIBLE_DEVICES"]= "0,1" # Select only GPUS 0 and 1
    print("Num Available GPUs: ", len(tf.config.list_physical_devices('GPU')))

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-partition', type=str, required=True,
                        help='Target KB: hp, go_bp, medic, ctd_chem, \
                        ctd_anat, chebi')
    parser.add_argument('-nil_entity', type=str, required=True,
                        help='The NIL entity to be linked to the chosen KB')
    parser.add_argument('-top_k', type=int, required=True,
                        help='The top-k candidates to return')
    
    args = parser.parse_args()

    model = load_model(args.partition, args.top_k)
    output = model.prediction(args.nil_entity)

    # Format output candidates
    cand_str = str()

    for cand in output:
        cand_str += cand[0] + '\t' + cand[1] + '\n'

    print('-----> Top ' + str(args.top_k) + ' candidates for "' 
          + args.nil_entity + '":\n'  
          + cand_str[:-1]) 