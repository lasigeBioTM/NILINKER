import argparse
import csv
import logging
import numpy as np
import os 
import sys
import tensorflow as tf
import tensorflow_addons as tfa
from src.NILINKER.nilinker import Nilinker
from src.utils.utils import retrieve_annotations_into_arrays, get_wc_embeds, get_kb_data
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
sys.path.append('./')


class SoftmaxCrossEntropyLoss(tf.keras.losses.Loss):
    """Adaptation of softmax cross entropy loss for batch data.
       It also takes into account class weighting. See
       https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
    """

    def __init__(self, name=None, weights=None, num_classes=None, 
                 test=False, batch_size=None):

        super().__init__(name=name)
        self.weights=weights
        self.num_classes=num_classes
        self.test=test
        self.batch_size = batch_size

    def call(self, y_batch, y_pred):
        """Calculates average loss of given batch data.

        :param y_batch: [description]
        :type y_batch: Tensorflow tensor
        :param y_pred: [description]
        :type y_pred: Tensorflow tensor
        
        :return: average_batch_loss
        :rtype: Tensorflow 'EagerTensor' object
        """
        # To calculate batch loss without class weighting
        # Uncomment the following two lines
        #y_true = tf.constant(y_true)
        #loss_item = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #               y_true, y_pred)#, axis=-1)
        
        y_true_tmp = tf.keras.utils.to_categorical(y_batch, 
            num_classes = self.num_classes, dtype='float32')
        y_true = tf.convert_to_tensor(y_true_tmp, dtype=tf.float32)
        
        # Losses for all instances in the batch
        loss_item = tf.nn.softmax_cross_entropy_with_logits(y_true,
                                                            y_pred, axis=-1) 
        
        if self.test: 
            # It is not necessary class weighting if 
            # we are testing the model
            average_batch_loss = tf.reduce_mean(loss_item) 
            
            return average_batch_loss

        else:
            # Apply class weigthing if training the model
            y_batch_list_tmp = y_batch.numpy().tolist()
            y_batch_list = [value[0] for value in y_batch_list_tmp]
            
            weights_batch_array = [self.weights[y_l] for y_l in y_batch_list]             
            weights_batch_tensor = tf.convert_to_tensor(
                                        weights_batch_array, 
                                        dtype=tf.float32)
            weights_batch_tensor = tf.reshape(
                                        weights_batch_array, 
                                        tf.shape(loss_item))
            weights_batch_tensor = tf.cast(
                                        weights_batch_tensor, 
                                        dtype=tf.float32)
      
            # Multiply loss of each item in batch by 
            # the respective class weight
            weighted_loss_item = tf.math.multiply(loss_item, 
                                                  weights_batch_tensor, 
                                                  name=None)
            
            average_batch_loss = tf.reduce_mean(weighted_loss_item) 
            
            return average_batch_loss


def train(args):
    """Trains NILINKER model according to predefined mode, and evaluates it 
    after the training is finished.

    :param args: An ArgumentParser object filled with input arguments.
    :type args: ArgumentParser object
    
    :return: prints the results (loss, micro-f1, macro-f1) 
        of the evaluation after model training
    """

    logging.info('-----> Starting {} mode on {} partition'.\
        format(args.mode, args.partition))

    logging.info('-----> Pre-processing KB-related data...')  
       
    word_embeds, candidate_embeds, \
        wc, embeds_words2id = get_wc_embeds(args.partition)
    
    params = [200, wc.candidate_num, args.top_k]
    kb_data = get_kb_data(args.partition)
    
    logging.info('-----> Retrieving annotations arrays...')
    x, y = retrieve_annotations_into_arrays(args.partition)
    
    #--------------------------------------------------------------------
    def train_on_split(x_train, x_test, y_train, y_test, 
                       split="", mode="final"):        
        """Trains NILINKER model in a single split, if mode is cross validation 
        there are more than 2 splits, otherwise just 1 split.
        
        :param x_train: input training data
        :type x_train: Numpy Array
        :param x_test: input test data
        :type x_test: Numpy array
        :param y_train: target training data
        :type y_train: Numpy array
        :param y_test: target test data
        :type y_test: Numpy array
        :param split: number of the current split if not final model, defaults 
            to ""
        :type split: str, optional
        :param mode: training model, it can be cross validation, final, or 
            optimization, defaults to "final"
        :type mode: str, optional
        :return: results, empty if final mode, otherwise includes evaluation 
            metrics
        :rtype: list
        """
        
        # Get number of unique classes in order to calculate 
        # each class weight. Since the dataset is imbalanced, we need to
        # use class weigthing to prevent model overfitting
        classes = np.unique(y_train) 
        class_weights_vect = compute_class_weight(
                                class_weight="balanced", 
                                classes=classes, 
                                y=y_train) 
        class_weight_dict = {classes[i] : class_weights_vect[i] 
                             for i in range(len(classes))}
       
        # Create a Dataset object for each set, it facilitates 
        # the handling of the dataset by the model
        train_dataset = tf.data.Dataset.from_tensor_slices(
                            (x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).\
                            batch(args.train_batch_size)
       
        test_dataset = tf.data.Dataset.from_tensor_slices(
                            (x_test, y_test))
        test_dataset = test_dataset.shuffle(buffer_size=1024).\
                            batch(args.test_batch_size)
        
        # Create Model 
        model = Nilinker(word_embeds, candidate_embeds, 
                         params, wc, kb_data, embeds_words2id)
        
        # Set evaluation metrics
        #micro_f1 = tfa.metrics.F1Score(
        #    num_classes=wc.candidate_num, average="micro", name="micro_f1")
        #macro_f1 = tfa.metrics.F1Score(
        #    num_classes=wc.candidate_num, average="macro", name="macro_f1")
        #mcc = tfa.metrics.MatthewsCorrelationCoefficient(
        #    num_classes=wc.candidate_num, name='mcc')
        #conf_matrix = tfa.metrics.MultiLabelConfusionMatrix(
        # num_classes=wc.candidate_num)

        top_1_acc = tf.keras.metrics.\
            TopKCategoricalAccuracy(k=1, name='Top_1_acc')
        top_2_acc = tf.keras.metrics.\
            TopKCategoricalAccuracy(k=2, name='Top_2_acc')
        top_3_acc = tf.keras.metrics.\
            TopKCategoricalAccuracy(k=3, name='Top_3_acc')
        top_4_acc = tf.keras.metrics.\
            TopKCategoricalAccuracy(k=4, name='Top_4_acc')
        top_5_acc = tf.keras.metrics.\
            TopKCategoricalAccuracy(k=5, name='Top_5_acc')
        top_10_acc = tf.keras.metrics.\
            TopKCategoricalAccuracy(k=10, name='Top_10_acc')
        top_20_acc = tf.keras.metrics.\
            TopKCategoricalAccuracy(k=20, name='Top_20_acc')

        logging.info('-----> Compiling model...')

        if args.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(
                            learning_rate=args.learning_rate)
        
        elif args.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                            learning_rate=args.learning_rate)
        
        model.compile(optimizer=optimizer,
                      loss = SoftmaxCrossEntropyLoss(name="train_loss", 
                      weights=class_weight_dict, 
                      num_classes=len(wc.candidate2id), 
                      test=False, batch_size=args.train_batch_size),                             
                      metrics=[top_1_acc, top_2_acc, top_3_acc, top_4_acc,
                               top_5_acc, top_10_acc, top_20_acc], 
                      run_eagerly=True)

        model_callbacks = list()
        model_filepath = str()

        if mode == "final":
            # After k-fold cross validation is done, 
            # we want to train the final version of the model
            # It is necessary to configure callbacks to save best model
            # Also, configure the criteria for interrupting the training
            model_filepath = 'data/nilinker_files/'+ args.partition + '/final/best.h5'
            model_callbacks = [
                        tf.keras.callbacks.EarlyStopping(
                            patience=args.patience,
                            # Stop training if loss is no longer improving
                            monitor="loss",
                            restore_best_weights=True),
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=model_filepath,
                            # Only saves the checkpoint if loss has improved
                            monitor="loss",
                            mode="min",
                            save_weights_only=True,
                            save_best_only=True,
                            verbose=0),
                              ]

        elif mode == "cross_valid":
            model_filepath = 'data/nilinker_files/' + args.partition \
                             + '/' + split + '/best.h5'
            model_callbacks = [
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=model_filepath,
                            # Only saves the checkpoint if loss has improved
                            monitor="loss",
                            mode="min",
                            save_weights_only=True,
                            save_best_only=True,
                            verbose=0)]

        if not os.path.exists(model_filepath[:-8]) and mode != "optimization":
            os.mkdir(model_filepath[:-8])

        # Train the model on train set and use reserved portion 
        # of train set as valid set in the end of epoch
        logging.info('-----> Fitting model...')
        
        history = model.fit(train_dataset, 
                            validation_data=None,
                            epochs=args.epochs, 
                            verbose=1,
                            callbacks=model_callbacks)
        
        results = list()

        if True:
            logging.info('-----> Evaluating...')
            
            results = model.evaluate(
                test_dataset, batch_size=args.test_batch_size, 
                verbose=1, return_dict=True) 

        return results
    #--------------------------------------------------------------------
    
    # If perfoming optimization, it is only necessary one split,
    # with a single train set and a signle test set.
    # If performing 5-fold cross validation,
    # 5 different train and test sets are sucessively generated. 
    # The final result is the average of the 5 results
    # If mode == "final", we only want the model file
    # So evaluation is not necessary
    # See https://scikit-learn.org/stable/modules/cross_validation.html
    
    logging.info('-----> Splitting the dataset...')  
    
    if args.mode == "final" or args.mode == "optimization": 
        # Define seed of random state for reproducible output
        test_size = float()

        if  args.mode == "final":
            test_size = 0.01
        
        elif args.mode == "optimization":
            test_size = 0.20

        x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                            test_size=test_size, 
                                                            random_state=10)
  
        # There is only 1 split, so 1 train set and 1 test set
        logging.info('-----> Training...')
        
        results = train_on_split(x_train, x_test, 
                                 y_train, y_test, 
                                 mode=args.mode)

        # Output results in csv file
        if args.mode == 'optimization':
            results_filename = 'logs/{}/opt/run_{}.csv'.format(args.partition, 
                                                               args.run_n)

            with open(results_filename, 'w') as results_file:
                writer = csv.writer(results_file)
                
                for key, value in results.items():
                    writer.writerow([key, value])
        
                results_file.close()

        logging.info('-----> Final results', results)
        print('-----> Final results:', results)

    else: 
        # To perform k-fold cross validation
        # Define seed of random state for reproducible output 
        kf = KFold(n_splits=args.num_fold, shuffle=True, random_state=50)  
        
        split_count = int()
        all_results = list()

        # There is at least 2 splits, so 2 train sets and 2 test sets
        # It is necessary to train and evaluate on each split sequentially
        for train_index, test_index in kf.split(x, y=y): 
            split_count += 1

            x_train, x_test = x[train_index], x[test_index] 
            y_train, y_test = y[train_index], y[test_index]
            
            logging.info(' -----> Training on split {}'.\
                format(str(split_count)))
            
            split_results = train_on_split(x_train, x_test, 
                                           y_train, y_test, 
                                           split=str(split_count), 
                                           mode=args.mode)
            
            all_results.append(split_results)
            
            logging.info('-----> Results Split {}: {}'.\
                format(str(split_count), split_results))
            
            print('-----> Results Split {}: {}'.\
                format(str(split_count), split_results))
        
        # Output results to csv file
        results_filename = 'logs/{}/cv.csv'.format(args.partition)

        with open(results_filename, 'w') as results_file:
            writer = csv.writer(results_file)
            
            for split in all_results:
            
                for key, value in split.items():
                    writer.writerow([key, value])
    
            results_file.close()

        logging.info('-----> Final results:', all_results)
        print('-----> Final results:', all_results)


if __name__ == "__main__":

    # Set the logging mode
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(3)

    # Info about available gpus
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # Select all available gpus
    #os.environ["CUDA_VISIBLE_DEVICES"]= "0,1" # Select only GPUS 0 and 1
    print("Num Available GPUs: ", len(tf.config.list_physical_devices('GPU')))

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, required=True,
                        help='mode of training: cross_valid, final, \
                             or optimization')
    parser.add_argument('-partition', type=str, required=True, 
                        help='Target KB, i.e., which annotations are used \
                              to train and evaluate the model: \
                              hp, go_bp, medic, ctd_chem, ctd_anat, chebi')     
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--train_batch_size', type=int, default=26)
    parser.add_argument('--test_batch_size', type=int, default=26)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--num_fold', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=1,
                        help='The top-k candidates to return')
   
    args = parser.parse_args()
    
    log_dir = './logs/{}/'.format(args.partition)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    log_filename = str()
    
    if args.mode == "cross_valid":
        args.num_fold = 5
     
        log_filename = log_dir + 'cross_valid.log'
    
    elif args.mode == "final":
        log_filename = log_dir + 'final.log'

    logging.basicConfig(
        filename=log_filename, level=logging.INFO, 
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
        filemode='w')

    train(args)