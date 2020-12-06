# EDCNN: Genome-Wide RNA-Binding Event Identification Using Evolutionary Deep Convolutional Neural Network
<br>
RNA binding proteins (RBPs) are a group of proteins associated with RNA regulation and metabolism, which play
an essential role in mediating the maturation, transport, localization, and RNA translation. Recently, Genome-wide RNA-binding event
detection methods have been developed for predicting RBPs. Unfortunately, those computational methods usually suffer from
limitations, such as high-dimensionality, data sparsity and model performance. To address those challenges, we propose evolutionary
deep convolution neural networks (EDCNNs) to identify protein-RNA interactions by synergizing evolutionary optimization with gradient
descent to optimize deep conventional neural network. In particular, EDCNN combines evolutionary algorithms and different gradient
descent models in a framework as a complementary algorithm, where the gradient descent step and the evolution step can alternately
optimize RNA binding events. To demonstrate the effectiveness of EDCNN, 24 large-scale CLIP-seq datasets are employed to
measure their performance. The experimental results reveal that EDCNN can provide superior performance against other
state-of-the-art methods. Furthermore, time complexity analysis, parameter analysis, motif analysis are conducted to demonstrate the
roubstness of our proposed algorithm from different perspectives.
 <br> <br>
 
# Dependency:
python 3.6 <br>
PyTorch 1.5.1 (http://pytorch.org/ ) <br>
Sklearn (https://github.com/scikit-learn/scikit-learn)


# Data 
Download the trainig and testing data from http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2 and decompress it in current dir. It has 24 experiments of 21 RBPs, and we need train one model per experiment.

# Usage:
python edcnn.py [-h] [--posi <postive_sequecne_file>] <br>
                 [--nega <negative_sequecne_file>] [--model_type MODEL_TYPE] <br>
                 [--out_file OUT_FILE] [--motif MOTIF] [--train TRAIN] <br>
                 [--model_file MODEL_FILE] [--predict PREDICT] [--motif_dir MOTIF_DIR]<br>
                 [--testfile TESTFILE] [--maxsize MAXSIZE] [--channel CHANNEL] <br>
                 [--window_size WINDOW_SIZE] [--local LOCAL] [--glob GLOB] <br>
                 [--ensemble ENSEMBLE] [--batch_size BATCH_SIZE] <br>
                 [--num_filters NUM_FILTERS] [--n_epochs N_EPOCHS] <br>
                 
# Use case:
Take ALKBH5 as an example, if you want to predict the binding sites for RBP ALKBH5 using ensembling local and global CNNs, and the default model is ensembling model. <br>
You first need train the model for RBP ALKBH5, then the trained model is used to predict binding probability of this RBP for your sequences. The follwoing CLI will train a ensembling model using local and global CNNs, which are trained using positves and negatives derived from CLIP-seq. <br>
# step 1:
1. python ideepe.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa --nega=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa --model_type=CNN --model_file=model.pkl --train=True 
<br>
For ensembling models, it will save 'model.pkl.local' and 'model.pkl.global' for local and global CNNs, respectively.<br>

# step 2:
2. python ideepe.py --testfile=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa --model_type=CNN --model_file=model.pkl --predict=True 
<br>

testfile is your input fasta sequences file, and the predicted outputs for all sequences will be defaulted saved in "prediction.txt". The value in each line corresponds to the probability of being RBP binding site for the sequence in fasta file. NOTE:if you have positive and negative sequecnes, please put them in the same sequecne file, which is fed into model for prediciton. DO NOT predict probability for positive and negative sequence seperately in two fasta files, then combine the prediction.

# Identify motifs:
You need install WebLogo (http://weblogo.berkeley.edu/) and TOMTOM in MEME Suite(http://meme-suite.org/doc/download.html?man_type=web) to search identifyed motifs against known motifs of RBPs. And also you need has positive and negative sequences when using motif option. <br> 
<br>
# step 3:
3. python ideepe.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa --nega=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa --model_type=CNN --model_file=model.pkl --motif=True --motif_dir=motifs

The identified motifs (PWMs, and Weblogo) are saved to be defaulted dir motifs (you can also use --motif_dir to configure your dir for motifs), and also include the report from TOMTOM.

# NOTE
When you train iDeepE on your own constructed benchmark dataset, if the training loss cannot converge, may other optimization methods, like SGD or RMSprop can be used to replace Adam in the code. 

# Contact
yaweiwang : wangyw19@mails.jlu.edu.cn
