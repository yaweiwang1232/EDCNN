## 
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
