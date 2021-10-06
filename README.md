# EDCNN:Identification of Genome-Wide RNA-binding Proteins Using Evolutionary Deep Convolutional Neural Network
<br>
RNA binding proteins (RBPs) are a group of proteins associated with RNA regulation and
metabolism, and play an essential role in mediating the maturation, transport, localization and translation
of RNA. Recently, Genome-wide RNA-binding event detection methods have been developed to predict
RBPs. Unfortunately, the existing computational methods usually suffer some limitations, such as highdimensionality, data sparsity and low model performance.
Deep convolution neural network has a useful advantage for solving high-dimensional and
sparse data. To improve further the performance of deep convolution neural network, we propose
evolutionary deep convolutional neural network (EDCNN) to identify protein-RNA interactions by
synergizing evolutionary optimization with gradient descent to enhance deep conventional neural network.
In particular, EDCNN combines evolutionary algorithms and different gradient descent models in a
complementary algorithm, where the gradient descent and evolution steps can alternately optimize the
RNA-binding event search. To validate the performance of EDCNN, an experiment is conducted on two
large-scale CLIP-seq datasets, and results reveal that EDCNN provides superior performance to other
state-of-the-art methods. Furthermore, time complexity analysis, parameter analysis and motif analysis
are conducted to demonstrate the effectiveness of our proposed algorithm from several perspectives.

<br>
<br>
In this package, we provides resources including: source codes, datasets, the EDCNN models, and usage examples. <br>
The flowcharts of identification of genome-wide RNA-binding proteins using evolutionary deep convolutional neural network is as follows:<br>

![Image text](https://github.com/yaweiwang1232/EDCNN/blob/main/Architecture.png)

## Requirements:
EDCNN is written in Python3 and requires the following dependencies to be installed: <br>
+ [PyTorch 1.8.1](http://pytorch.org/) <br>
+ [Sklearn](https://github.com/scikit-learn/scikit-learn)
+ [matplotlib 3.3.4](https://matplotlib.org/)
+ [numpy 1.16.2](http://numpy.org/)
+ [deap 1.3.1](https://github.com/deap/deap)
+ [weblogo 3.7.8](http://weblogo.berkeley.edu/)

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
The training and testing data of RBP47_Dataset, protein binding microarrays(PBM) data, and eCLIP data can be found in this Repositories.<br>
The available eCLIP data were downloaded from ENCODE (Davis et al., 2018), containing 93 diverse RBPs in HepG2 and K562 cells.<br>
All data has sequences and labels, and you can also get it from https://doi.org/10.6084/m9.figshare.16746514.v2
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

### 1.Train step:

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

### 2.Predict step:
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
### 3.Motif step:
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
run_edcnn | Implementation process of EDCNN
train_network | Optimizing neural network parameters using different optimizers in train step
generate_offspring | Adding Gaussian noise to neural network model parameters to generate offspring
individual_evaluation | Evaluating individuals in population and output fitness
predict_network | Predicting using the best model

## Contact
For questions, comments and concerns, please contact
Yawei Wang(wangyw19@mails.jlu.edu.cn), Yuning Yang, and Xiangtao Li.
