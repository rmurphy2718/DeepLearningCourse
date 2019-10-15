# Course Project: Variational Inference in Convolutional Neural Networks

A Bayesian approach to training neural networks can come with benefits such
as natural weight regularization, straightforward approaches for weight pruning,
and making prediction more uncertain in regions with sparse training data. Previous work showed that feedforward networks could be trained in the Bayesian
framework via a backpropagation strategy (Bayes by Backprop) and demonstrated
the aforementioned benefits experimentally. A recent paper extended the method
to Convolutional Neural Networks but did not explore whether said advantages
carry over to CNNs. We evaluated them here.  

Bayes by Backprop is explained in the ICML 2015 paper [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424).
We modify the GitHub code provided by the authors of [Bayesian Convolutional Neural Networks](https://arxiv.org/abs/1806.05978v4), which was available on [GitHub](https://github.com/felix-laumann/Bayesian_CNN) but has been removed. 
	
(Fall 2018)

### Authors

Ryan L. Murphy, Jincheng Bai.  

### Code
We use PyTorch version 0.4.1.
 