# Preparation

Guide to reproduce all the work from scratch.

---------------------------------------------------------

## Summary
- [1. EvaNIL](#1)
  - [1.1. Download the EvaNIL dataset](#1.1)
  - [1.2. Generate EvaNIL dataset from source corpora (optional)](#1.2)
- [2. Prepare BioSyn](#2)
  - [2.1. Get the modified version of the repository](#2.1)
  - [2.2. Setup](#2.2)
  - [2.3. Retrieve the already preprocessed EvaNIL files](#2.3)
  - [2.4. Get the already trained models](#2.4)
  - [2.5. Train BioSyn models on the EvaNIL dataset (optional)](#2.5)
  - [2.6. Get the Named Entity Linking datasets](#2.6)
- [3. NILINKER](#3)
  - [3.1. Preparing NILINKER (optional)](#3.1)
  - [3.2. Train NILINKER models (optional)](#3.2)
    - [3.2.1. Hyperparameter optimization](#3.2.1)
    - [3.2.2. Final training](#3.2.2)
- [4. Named Entity Linking Evaluation datasets](#4)
  - [4.1. Preprocess BC5CDR-Disease, BC5CDR-Chemical, NCBI Disease datasets (optional)](#4.1)
  - [4.2. Preprocess the CHR, GSC+ and PHAEDRA datasets (optional)](#4.2)
  - [4.3. Download the already preprocessed datasets](#4.3)
  

---------------------------------------------------------


# 1. EvaNIL<a name="1"></a>

## 1.1. Download the EvaNIL dataset<a name="1.1."></a>
To download the ready EvaNIL dataset run:

```
wget
tar -xvf
rm
```

## 1.2. Generate EvaNIL dataset from source corpora (optional) <a name="1.2."></a>

If you want generate yourself the EvaNIL dataset from scratch first get the necessary data:

Then run:

```
./get_EvaNIL_prepartion_data.sh
python src/evanil/dataset.py -partition <partition>
```

Arg
 - partition: "medic" (MEDIC), "ctd_chem" (CTD-Chemicals), "ctd_anat" (CTD-Anatomy), "chebi" (ChEBI),"go_bp" (Gene Ontology - Biological Process), "hp" (Human Phenotype Ontology)

Files will be in the directory 'data/evanil/'.

---------------------------------------------------------

# 2. Prepare BioSyn<a name="2"></a>

## 2.1. Get the modified version of the repository<a name="2.1"></a>

```
wget git clone https://github.com/pedroruas18/BioSyn.git
cd BioSyn/
```

## 2.2. Setup<a name="2.2"></a>

- CONDA: First is necessary to install conda:
   
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
```

Type yes

Go to root directory and run:

```
cd ../../root/
source ~/.bashrc
cd ../../NILINKER/BioSyn/
```

- Install the necessary requirements:

```
conda create -n BioSyn python=3.7
conda activate BioSyn
conda install numpy tqdm scikit-learn
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install transformers==4.18.0
conda install pandas
conda install gdown

```

*Note: modify cudatoolkit version according to your setup

- Setup the correct path:

```
export PYTHONPATH="${PYTHONPATH}:"
```

- Create directory 'datasets'

```
mkdir datasets
```


## 2.3. Retrieve the already preprocessed EvaNIL files<a name="2.3"></a>
To train the BioSyn models on the EvaNIL datasets or to use them for inference in the NEL datasets it is necessary to download them:

```
wget
tar -xvf evanil_preprocessed_biosyn.tar.gz
rm ''
```

## 2.4. Get the already trained models<a name="2.4"></a>

Retrieve the already trained models:

```
mkdir tmp/
cd tmp/
wget
cd ../ 
```

## 2.5. Train BioSyn models on the EvaNIL dataset (optional)<a name="2.5"></a>
Or instead, if you want to train the BioSyn models from scratch on the EvaNIL dataset:

```
MODEL_NAME_OR_PATH=dmis-lab/biobert-base-cased-v1.1
OUTPUT_DIR=./tmp/biosyn-biobert-<partition>/
DATA_DIR=./datasets/evanil/<partition>/

CUDA_VISIBLE_DEVICES=0 python train.py --model_name_or_path ${MODEL_NAME_OR_PATH} --train_dictionary_path ${DATA_DIR}/preprocessed/train_dictionary.txt --train_dir ${DATA_DIR}/preprocessed/processed_train --output_dir ${OUTPUT_DIR} --use_cuda --topk 1 --epoch 2 --train_batch_size 16 --learning_rate 1e-5 --max_length 25
```

## 2.6. Get the Named Entity Linking datasets<a name="2.6"></a>

It is necessary to download the BC5CDR-Disease, BC5CDR-Chemical and the NCBI-Disease datasets 


```
cd datasets/

# BC5CDR-Disease
gdown https://drive.google.com/uc?id=1moAqukbrdpAPseJc3UELEY6NLcNk22AA
tar -xvf bc5cdr-disease.tar.gz

# BC5CDR-Chemical
gdown  https://drive.google.com/uc?id=1mgQhjAjpqWLCkoxIreLnNBYcvjdsSoGi
tar -xvf bc5cdr-chemical.tar.gz

# NCBI-Disease
gdown https://drive.google.com/uc?id=1mmV7p33E1iF32RzAET3MsLHPz1PiF9vc
tar -xvf ncbi-disease.tar.gz

cd ../
```

---------------------------------------------------------

# 3. NILINKER<a name="3"></a>

## 3.1. Preparing NILINKER (optional)<a name="3.1"></a>

You can download the Word-Concept dictionaries, the embeddings and the annotations 
files used in the experiments.

However, if you want to generate yourself those files that are associated with a given partition
of the EvaNIL dataset, run:

```
./get_NILINKER_preparation_data.sh
./prepare_NILINKER.sh <partition>
```

Arg:
- partition: 'medic', 'ctd_anat', 'ctd_chem', 'chebi', 'go_bp' or 'hp'


At this stage NILINKER is ready for training or hyperparameter optimization


## 3.2. Train NILINKER models (optional)<a name="3.2"></a>

### 3.2.1. Hyperparameter optimization<a name="3.2.1"></a>

Run experiments to find best combination of hyperparameters:

```
python src/NILINKER/hyperparameter_optimization.py -partition <partition>
```

Args:
  - partition: 'medic', 'ctd_anat', 'ctd_chem', 'chebi', 'go_bp' or 'hp'


### 3.2.2. Final training<a name="3.2.2"></a>

To train the final version of the NILINKER model in given EvaNIL partition, use the same script but change the value of the arg 'mode' to 'final'.

Example:

```
python src/NILINKER/train_nilinker.py -mode train -partition chebi
```

The file associated with the trained model 'best.h5' will in the directory 'data/nilinker_files/chebi/final/'.

---------------------------------------------------------
# 4. Named Entity Linking Evaluation datasets<a name="4"></a>

Datasets (with targe Knowledge Bases within parentheses):
  - BC5CDR-Disease (MEDIC vocabulary)
  - BC5CDR-Chemical (CTD-Chemical vocabulary)
  - NCBI Disease corpus (MEDIC vocabulary)
  - GSC+ (Human Phenotype Ontology)
  - CHR (ChEBI ontology)
  - PHAEDRA (CTD-Chemical vocabulary)


## 4.1. Preprocess BC5CDR-Disease, BC5CDR-Chemical, NCBI Disease datasets (optional)<a name="4.1"></a>
Get the modified version of the repository "Fair Evaluation in Concept Normalization: a Large-scale Comparative Analysis for BERT-based Models"

```
git clone https://github.com/pedroruas18/Fair-Evaluation-BERT.git
```

Install the requirements:

```
pip install -r requirements.txt
```

Inside the repository, execute the script 'prepare.sh':

```
cd Fair-Evaluation-BERT
chmod +x prepare.sh
./prepare.sh
cd ../
```

This will prepare the provided datasets (BC5CDR-Disease, BC5CDR-Chemical, NCBI Disease) to be used with REEL-based models.



## 4.2. Preprocess the CHR, GSC+ and PHAEDRA datasets (optional)<a name="4.2"></a>

First get the data:

```
chmod +x get_NEL_evaluation_data.sh
./get_NEL_evaluation_data.sh
```

And then preprocess it to prepare it to be used with REEL-based models:

```
cd src/nel_evaluation/
python process_nel_corpora.py gsc+
python process_nel_corpora.py phaedra
python process_nel_corpora.py chr
cd ../../
```

## 4.3. Download the already preprocessed datasets<a name="4.3"></a>

Run:

```
wget 
tar -xvf
rm
```
