**3.1.** The UC Irvine machine learning data repository hosts a famous collection of
data on whether a patient has diabetes (the Pima Indians dataset), originally
owned by the National Institute of Diabetes and Digestive and Kidney Diseases
and donated by Vincent Sigillito. This can be found at http://archive.ics.uci.
edu/ml/datasets/Pima+Indians+Diabetes. This data has a set of attributes of
patients, and a categorical variable telling whether the patient is diabetic or
not. For several attributes in this data set, a value of 0 may indicate a missing
value of the variable.  

**(a)** Build a simple naive Bayes classifier to classify this data set. You should
hold out 20% of the data for evaluation, and use the other 80% for training.
You should use a normal distribution to model each of the class-conditional
distributions. You should write this classifier yourself (it’s quite straightforward),
but you may find the function createDataPartition in the R
package caret helpful to get the random partition.

**(b)** Now adjust your code so that, for attribute 3 (Diastolic blood pressure),
attribute 4 (Triceps skin fold thickness), attribute 6 (Body mass index),
and attribute 8 (Age), it regards a value of 0 as a missing value when
estimating the class-conditional distributions, and the posterior. R uses
a special number NA to flag a missing value. Most functions handle this
number in special, but sensible, ways; but you’ll need to do a bit of looking
at manuals to check. Does this affect the accuracy of your classifier?  

**(c)** Now use the caret and klaR packages to build a naive bayes classifier
for this data, assuming that no attribute has a missing value. The caret
package does cross-validation (look at train) and can be used to hold out
data. The klaR package can estimate class-conditional densities using a
density estimation procedure that I will describe much later in the course.
Use the cross-validation mechanisms in caret to estimate the accuracy of
your classifier. I have not been able to persuade the combination of caret
and klaR to handle missing values the way I’d like them to, but that may
be ignorance (look at the na.action argument).  

**(d)** Now install SVMLight, which you can find at http://svmlight.joachims.
org, via the interface in klaR (look for svmlight in the manual) to train
and evaluate an SVM to classify this data. You don’t need to understand
much about SVM’s to do this — we’ll do that in following exercises. You
should hold out 20% of the data for evaluation, and use the other 80% for
training. You should NOT substitute NA values for zeros for attributes 3,
4, 6, and 8.  

**3.3.** The UC Irvine machine learning data repository hosts a collection of data on
heart disease. The data was collected and supplied by Andras Janosi, M.D., of
the Hungarian Institute of Cardiology, Budapest; William Steinbrunn, M.D.,
of the University Hospital, Zurich, Switzerland; Matthias Pfisterer, M.D., of
the University Hospital, Basel, Switzerland; and Robert Detrano, M.D., Ph.D.,
of the V.A. Medical Center, Long Beach and Cleveland Clinic Foundation. You
can find this data at https://archive.ics.uci.edu/ml/datasets/Heart+Disease.
Use the processed Cleveland dataset, where there are a total of 303 instances
with 14 attributes each. The irrelevant attributes described in the text have
been removed in these. The 14’th attribute is the disease diagnosis. There are
records with missing attributes, and you should drop these.  

**(a)** Take the disease attribute, and quantize this into two classes, num = 0
and num > 0. Build and evaluate a naive bayes classifier that predicts
the class from all other attributes Estimate accuracy by cross-validation.
You should use at least 10 folds, excluding 15% of the data at random to
serve as test data, and average the accuracy over those folds. Report the
mean and standard deviation of the accuracy over the folds.  

**(b)** Now revise your classifier to predict each of the possible values of the
disease attribute (0-4 as I recall). Estimate accuracy by cross-validation.
You should use at least 10 folds, excluding 15% of the data at random to
serve as test data, and average the accuracy over those folds. Report the
mean and standard deviation of the accuracy over the folds
