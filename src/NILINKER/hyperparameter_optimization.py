import argparse
import logging
import os
import tensorflow as tf
from train_nilinker import train


if __name__ == "__main__":

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Info about available gpus
    #tf.get_logger().setLevel('INFO')
    #tf.autograph.set_verbosity(0, alsologtostdout=True)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
    print('Num Available GPUs: ', len(tf.config.list_physical_devices('GPU')))

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-partition', type=str, required=True, 
                        help='Annotations to train and evaluate the model: \
                        hp, go_bp, medic, ctd_chem, ctd_anat, chebi')          
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--top_k', type=int, default=1,
                        help='The top-k candidates to return')

    args = parser.parse_args()
    args.mode = 'optimization'

    log_dir = './logs/{}/opt/'.format(args.partition)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_filename = log_dir + 'optimization.log'
    logging.basicConfig(
        filename=log_filename, level=logging.INFO, 
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
        filemode='w')
    
    # Run experiments to find best combination of hyperparameters
    run_n = 0
    HP_OPTIMIZER = ['adam', 'sgd']
    HP_LEARNING_RATE =  [0.00001, 0.0001, 0.001, 0.01, 0.1]

    for optimizer in HP_OPTIMIZER:
        
        for learning_rate in HP_LEARNING_RATE:         
            args.optimizer = optimizer
            args.learning_rate = learning_rate
            
            run_name = "run-%d" % run_n
            args.run_n = run_n

            logging.info('---- Starting trial: %s' % run_name)
            logging.info(args)
            
            train(args)
            
            run_n += 1