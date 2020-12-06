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
Take ALKBH5 as an example, if you want to predict the binding sites for RBP ALKBH5 using local and global CNNs <br>
# step 1:
1. python3 edcnn.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa --nega=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa --model_type=CNN --model_file=model.pkl --train=True 
<br>
Our propose EDCNN will save 'best.model.pkl.local' and 'best.model.pkl.global' for local and global CNNs, respectively.<br>

# step 2:
2. python3 edcnn.py --testfile=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa --model_type=CNN --model_file=model.pkl --predict=True 
<br>
predict step

# Identify motifs:
You need install WebLogo (http://weblogo.berkeley.edu/) and TOMTOM in MEME Suite(http://meme-suite.org/doc/download.html?man_type=web) to search identifyed motifs against known motifs of RBPs. And also you need has positive and negative sequences when using motif option. <br> 
<br>
# step 3:
3. python3 edcnn.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa --model_type=CNN --model_file=model.pkl --motif=True --motif_dir=motifs
# Calculate the mean values of AUCs

The identified motifs (PWMs, and Weblogo) are saved to be defaulted dir motifs (you can also use --motif_dir to configure your dir for motifs), and also include the report from TOMTOM.

# Contact
yaweiwang : wangyw19@mails.jlu.edu.cn
