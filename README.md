# NILINKER

Attention-based approach to NIL entity linking.


```
P Ruas, FM Couto. NILINKER: attention-based approach to NIL entity linking
```

---------------------------------------------------------

## Summary
- [1. Setup](#1)
  - [1.1. Dockerfile](#1.1)
  - [1.2. Get data](#1.2)
- [2. Preparing NILINKER](#2)
- [3. Using NILINKER](#3)
  - [3.1. Hyperparameter optimization](#3.1)
  - [3.2. k-fold cross validation](#3.2)
  - [3.3. Final training](#3.3)
  - [3.4. Inference](#3.4)
- [4. Improving Named Entity Linking with NILINKER](#4)
  

---------------------------------------------------------

## 1. Setup<a name="1"></a>

### 1.1. Dockerfile<a name="1.1"></a>
The Dockerfile includes the commands to setup the appropriate environment.

In the root directory, build the image by running:

```
docker build .
```

Then run a container:

```
nvidia-docker run -v <root_directory>:/nil_linking/ --name <container_name> --gpus '"device=<>"' -it <image ID> bash  
```

Example with 1 available gpu:

```
nvidia-docker run -v nil_linking/:/nil_linking/ --name nilinker  --gpus '"device=1"' -it edb49f6a133b bash  
```

After running the container, run in the root directory:

```
export PYTHONPATH="${PYTHONPATH}:"
```

### 1.2. Get data<a name="1.2"></a>
To download the necessary data to reproduce the experiments and to use NILINKER:

```
./get_data.sh
```

This script will retrieve the necessary knowledge base, corpora, annotations and embeddings files, and the version of the EvaNIL dataset that was used. You can also manually access some of this [data](https://zenodo.org/record/5927300#.YffAyvvLdak).

The EvaNIL dataset can be used to train and evaluate models that perform NIL entity linking. 
In the experiments we used a slighlty modified version of the EvaNIL dataset.

It is possible to directly retrieve the [original version of the dataset](https://zenodo.org/record/5849231).

If you want instead generate yourself the dataset from scratch, run:

```
python src/evanil/dataset.py -partition <partition>
```

Arg
 - partition: "medic" (MEDIC), "ctd_chem" (CTD-Chemicals), "ctd_anat" (CTD-Anatomy), "chebi" (ChEBI),"go_bp" (Gene Ontology - Biological Process), "hp" (Human Phenotype Ontology)


---------------------------------------------------------

## 2. Preparing NILINKER (Optional)<a name="2"></a>

You can download the Word-Concept dictionaries, the embeddings and the annotations 
files used in the experiments by running the script 'get_data.sh'.

However, if you want to generate yourself those files that are associated with a given partition
of the EvaNIL dataset, run:

```
./prepare_NILINKER.sh <partition>
```

Arg:
- partition: 'medic', 'ctd_anat', 'ctd_chem', 'chebi', 'go_bp' or 'hp'


At this stage NILINKER is ready for training, k-fold-cross validation or hyperparameter optimization

---------------------------------------------------------

## 3. Using NILINKER<a name="3"></a>

### 3.1. Hyperparameter optimization<a name="3.1"></a>

Run experiments to find best combination of hyperparameters:

```
python src/NILINKER/hyperparameter_optimization.py -partition <partition>
```

Args:
  - partition: 'medic', 'ctd_anat', 'ctd_chem', 'chebi', 'go_bp' or 'hp'


### 3.2. k-fold cross validation<a name="3.2"></a>

To perform k-fold cross validation run the following command, setting the values for each argument according to the values obtained in the previous hyperparameter optimization step:

```
python src/NILINKER/train_nilinker.py -partition <partition> --num_fold <num_fold>
```

Args:

  -mode: training mode ('cross_valid', 'final', 'optimization')
  -partition: 'hp', 'go_bp', 'medic', 'ctd_chem', 'ctd_anat', 'chebi'     
  
  --epochs: number of training epochs (default=7)
  
  --train_batch_size: default=26
  
  --test_batch_size: default=26
  
  --learning_rate: default=0.01
  
  --patience: number of training epochs without decreasing the evaluation loss (default=5)
  
  --optimizer: default='adam'
  
  --num_fold: number of splits of k-fold cross validation (default=1)
  
  --top_k: The top-k candidates to return by the model (default=1)

Example:

```
python src/NILINKER/train_nilinker.py -mode cross_valid -partition chebi --num_fold 5
```

### 3.3. Final training<a name="3.3"></a>

To train the final version of the NILINKER model in given EvaNIL partition, use the same script but change the value of the arg 'mode' to 'final'.

Example

```
python src/NILINKER/train_nilinker.py -mode final -partition chebi
```

---------------------------------------------------------

### 3.4. Inference<a name="3.4"></a>

It is possible to use a previously trained NILINKER model for predicting the top candidates for a given NIl entity.

Example (run from root directory):

```
from src.NILINKER.predict_nilinker import load_model

target_kb = 'medic' 
top_k = 10 # Top-k candidates to return for input entity

nilinker = load_model(target_kb, top_k=top_k))

entity = "parkinsonian disorders"

top_candidates = nilinker.prediction(entity)
```

---------------------------------------------------------

## 4. Improving Named Entity Linking with NILINKER<a name="4"></a>
We also adapted [REEL](https://github.com/lasigeBioTM/REEL), a biomedical Named Entity Linking model, to be used jointly with NILINKER. Run the following command to apply this model to selected datasets:

```
./run_reel.sh <dataset> <candidate link mode> <NIL entity linking approach>
```

Args:
  - dataset: The target dataset containing the entities to link. Possible values: 'bc5cdr_medic', 'bc5cdr_chem', 'gsc+', 'ncbi_disease', 'chr', 'phaedra',or 'ChebiPatents'                                                   
  - candidate link mode: How to add edges in the disambigution graphs the REEL builds for each document. Possible values: 'kb' (only relations described in target knowledge base), 'corpus' (only relations extracted from target dataset),'corpus_kb' (knowledge base and corpus relations)                                 
  - NIL entity linking approach: Approach to deal with NIL entities. Possible values: 'none' (the original REEL models), 'StringMatcher', 'NILINKER'
