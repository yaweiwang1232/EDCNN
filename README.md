# EDCNN:Identification of Genome-Wide RNA-binding Proteins Using Evolutionary Deep Convolutional Neural Network
<br>
The tool is developed for identification of genome-wide RNA-binding proteins using evolutionary deep convolutional neural network.<br>
We propose
evolutionary deep convolutional neural network (EDCNN) to identify protein-RNA interactions by
synergizing evolutionary optimization with gradient descent to enhance deep conventional neural network.<br>
In this package, we provides resources including: source codes, datasets, the EDCNN models, and usage examples. <br>

The flowcharts of identification of genome-wide RNA-binding proteins using evolutionary deep convolutional neural network is as follows:<br>
![Image text](https://github.com/yaweiwang1232/EDCNN/blob/main/Architecture.png)
## Requirements:
EDCNN is written in Python3 and requires the following dependencies to be installed: <br>
+ [PyTorch](http://pytorch.org/) <br>
+ [Sklearn](https://github.com/scikit-learn/scikit-learn)
+ [deap](https://github.com/deap/deap)
+ [numpy](http://numpy.org/)
+ [weblogo](http://weblogo.berkeley.edu/)

## Installation
We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/). 
```
conda create -n edcnn python=3.6
conda activate edcnn
git clone https://github.com/yaweiwang1232/EDCNN.git
cd EDCNN
python3 -m pip install -r requirements.txt
```






## Data 
For RBP-24, the training and testing data can be downloaded from http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2 and decompress it in current dir. It has 24 experiments of 21 RBPs.<br>
The training and testing data of RBP47_Dataset, protein binding microarrays(PBM) data, and eclip data can be found in this Repositories.<br>
All data has sequences and labels, and you can also get it from https://doi.org/10.6084/m9.figshare.16746514.v1
easily.
## Usage:

```
python edcnn.py [-h] [--posi <postive_sequecne_file>]
                 [--nega <negative_sequecne_file>]
                 [--model_type MODEL_TYPE] 
                 [--population_size POPULATION_SIZE]
                 [--generation_number GENERATION_NUMBER]
                 [--out_file OUT_FILE]
                 [--motif MOTIF] 
                 [--train TRAIN] 
                 [--model_file MODEL_FILE] 
                 [--predict PREDICT]
                 [--motif_dir MOTIF_DIR]
                 [--batch_size BATCH_SIZE] 
                 [--num_filters NUM_FILTERS] 
```



                 
## Use case:

### 1.train step:

Train the model using postive_sequecne_file and negative_sequecne_file.<br>
```
python3 edcnn.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa \
                 --nega=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa \
                 --model_type=CNN \
                 --population_size=population_size \
                 --generation_number=generation_number \
                 --model_file=model.pkl \
                 --train=True 
                 
```
<br>
Our proposed EDCNN will autosave best.model.pkl.local and best.model.pkl.global for local and global CNNs. <br>

### 2.predict step:
predict the binding probability for your sequences using best.model.pkl.local and best.model.pkl.global.
```
python3 edcnn.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa \
                 --nega=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.negatives.fa \
                 --model_type=CNN \
                 --population_size=population_size \
                 --generation_number=generation_number\
                 --model_file=model.pkl \
                 --predict=True 
```
### 3.detect motif step:
Identify the binding sequence motifs using best.model.pkl.local and best.model.pkl.global.
```
python3 edcnn.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa \
                 --nega=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa \
                 --model_type=CNN \
                 --population_size=population_size \
                 --generation_number=generation_number \
                 --model_file=model.pkl \
                 --motif=True \
                 --motif_dir=motifs
```
--motif_dir:The saved dir for identified motif.
<br>
You need install WebLogo(http://weblogo.berkeley.edu/) and TOMTOM in MEME Suite(http://meme-suite.org/) to search identifyed motifs against known motifs of RBPs. <br> 
TOMTOM in MEME Suite can be found in https://github.com/yaweiwang1232/EDCNN/blob/main/meme_4.11.4.tar.gz.
<br>

We also give the key functions of the source code and their detailed description.<br>

Function | Description
---|---
run_edcnn(args) | Implementation process of EDCNN
train_network | Optimizing neural network parameters using different optimizers in train step
generate_offspring | Adding Gaussian noise to neural network model parameters to generate offspring
individual_evaluation | Evaluating individuals in population and output fitness
predict_network | Predicting using the best model

## Contact
For questions, comments and concerns, please contact
Yawei Wang(wangyw19@mails.jlu.edu.cn), Yuning Yang, and Xiangtao Li.
