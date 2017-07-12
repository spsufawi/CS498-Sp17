**3.6.** The UC Irvine machine learning data repository hosts a collection of data on
the whether p53 expression is active or inactive.
You can find this data at [https://archive.ics.uci.edu/ml/datasets/p53+Mutants](https://archive.ics.uci.edu/ml/datasets/p53+Mutants).
There are a total of 16772 instances, with 5409 attributes per instance. Attribute
5409 is the class attribute, which is either active or inactive. There are
several versions of this dataset. You should use the version K8.data.

**(a)** Train an SVM to classify this data, using stochastic gradient descent. You
will need to drop data items with missing values. You should estimate
a regularization constant using cross-validation, trying at least 3 values.
Your training method should touch at least 50% of the training set data.
You should produce an estimate of the accuracy of this classifier on held
out data consisting of 10% of the dataset, chosen at random.

**(b)** Now train a naive bayes classifier to classify this data. You should produce
an estimate of the accuracy of this classifier on held out data consisting
of 10% of the dataset, chosen at random.

**(c)** Compare your classifiers. Which one is better? why?
