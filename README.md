# NILINKER

Model to associate NIL (out-of-KB or unlinkable) entities with the best available entry in a target Knowledge Base.

Citation:

```
P Ruas, FM Couto. NILINKER: attention-based approach to NIL entity linking
```
---------------------------------------------------------

## Summary
- [1. Setup](#1)
- [2. Using NILINKER for inference](#2)
    - [2.1. Get data](#2.1)
    - [2.2. Inference](#2.2)
- [3. Evaluation](#3)
  - [3.1. Evaluation on the EvaNIL dataset](#3.1)
  - [3.2. Named Entity Linking evaluation](#3.2)
- [4. Preparation](#4)
  

---------------------------------------------------------

## 1. Setup<a name="1"></a>

The Dockerfile includes the commands to setup the appropriate environment.

In the root directory, build the image by running:

```
docker build . -t nilinker_image
```

Then run a container:

- Using the docker command:

```
docker run -v $(pwd):/NILINKER/ --name nilinker -it nilinker_image bash
```

- Using the nvidia-docker command:

```
nvidia-docker run -v <root_directory>:/nil_linking/ --name <container_name> --gpus '"device=<>"' -it <image ID>/<image tag> bash  
```

Example with 1 available gpu:

```
nvidia-docker run -v $(pwd):/NILINKER/ --name nilinker  --gpus '"device=1"' -it nilinker_image bash  
```

After running the container, run in the root directory of the repository:

```
export PYTHONPATH="${PYTHONPATH}:"
```


---------------------------------------------------------

## 2. Using NILINKER for inference<a name="2"></a>

You can use a previously trained NILINKER model for predicting the top-k candidates for a given NIL (unlinkable or out-of-KB) entity. The following models are available:

- NILINKER-MEDIC ([MEDIC vocabulary](http://ctdbase.org/voc.go?type=disease))
- NILINKER-CTD-Chem ([CTD Chemicals vocabulary](http://ctdbase.org/voc.go?type=chem))
- NILINKER-ChEBI ([Chemical Entities of Biological Interest (ChEBI) ontology](https://www.ebi.ac.uk/chebi/))
- NILINKER-CTD-ANAT ([CTD Anatomy vocabulary](http://ctdbase.org/voc.go?type=anatomy)
- NIKINKER-HP ([Human Phenotype Ontology]())
- NILINKER-GO-BP ([Gene Ontology-Biological Process](http://geneontology.org/))


### 2.1. Get data<a name="2.1"></a>
To download the necessary data to strictly use NILINKER for inference:

```
./get_NILINKER_use_data.sh
```


### 2.2. Inference<a name="2.2"></a>

Python script example (run from root directory previously defined):

```
from src.NILINKER.predict_nilinker import load_model

# Args
target_kb = 'medic' 
top_k = 10 # Top-k candidates to return for input entity

nilinker = load_model(target_kb, top_k=top_k))

entity = "parkinsonian disorders"

top_candidates = nilinker.prediction(entity)
```

Args:
- tarket_kb: the NIL entity mention will be linked to concepts of the selected target Knowlegde Base ('medic', 'ctd_chem', 'chebi', 'ctd_anat', 'hp' or 'go_bp')
- top_k: the number candidates to retrieve for the NIL entity mention (e.g. 1, 2, 5, 10, 20, ...)

---------------------------------------------------------

## 3. Evaluation<a name="3"></a>

To reproduce the evaluations described in the article follow the instructions below.

### 3.1. Evaluation on the EvaNIL dataset<a name="3.1"></a>

To obtain the EvaNIL dataset follow the instructions [here]() ('1. EvaNIL').

Models:
- StringMatcher
- [BioSyn](https://github.com/dmis-lab/BioSyn). To use this model follow the instructions [here]() ('2. Prepare BioSyn').
- NILINKER

Run the script adjusting the arguments:

```
python evaluation_evanil.py -partition <partition> -model <> --refined <refined>
```

Args:
- -partition: Evanil partition name ('medic', 'ctd_chem', 'chebi', 'ctd_anat', 'hp' or 'go_bp' )
- -model: model to evaluate ('stringMatcher', 'biosyn' or nilinker')
- --refined: True to evaluate model on refined version of the test set (test set exluding entity mentions that appear on training and development sets), False (by default) to evaluate on the original test set.

Example:

```
python evaluation_evanil.py -partition medic -model nilinker 
```

Output:

```
tp: 70998 
fp: 9052 
fn: 0
Results
Precision: 0.8869
Recall: 1.0000
F1-score: 0.9401
```

### 3.2. Named Entity Linking evaluation<a name="3.2"></a>

To obtain the necessary datasets follow the instructions [here]() ('4. Named Entity Linking Evaluation datasets').

Models:
- SapBERT-based BioSyn models: [biosyn-sapbert-bc5cdr-disease](https://huggingface.co/dmis-lab/biosyn-sapbert-bc5cdr-disease), [biosyn-sapbert-bc5cdr-chemical](https://huggingface.co/dmis-lab/biosyn-sapbert-bc5cdr-chemical), [biosyn-sapbert-ncbi-disease](https://huggingface.co/dmis-lab/biosyn-sapbert-ncbi-disease)
- [REEL](https://github.com/lasigeBioTM/REEL/blob/master/README.md)
- REEL-StringMatcher
- REEL-NILINKER

NOTE: To use BioSyn model follow the instructions [here]() ('2. Prepare BioSyn').

After preparing the datasets and the models, run the script:

```
python evaluation_nel.py -dataset <dataset> -model <model> -subset <subset> 
```

Args:
- -dataset: the datasets where the evaluation will be performed ('bc5cdr_medic', bc5cdr_chem', 'chr', 'gsc+', 'phaedra', 'ncbi_disease')
- -model: 'biosyn', 'reel', 'reel_stringMatcher', 'reel_nilinker'
- -subset: 'test' or 'test_refined' 

Example:

```
python evaluation_nel.py -dataset bc5cdr_medic -model reel_nilinker -subset test
```

Output:

```
Number of documents: 500
Total entities (NIL+non-NIL): 3287
True NIL entities: 0
True Positives: 3287
False Positives: 1000
Accuracy: 0.7667
```

## 4. Preparation<a name="4"></a>
To reproduce all the steps performed to build the EvaNIl dataset, to train the NILINKER and BioSyn models, and to preprocess the Named Entity Linking datasets follow the [instructions]().