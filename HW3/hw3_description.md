**4.10.**
CIFAR-10 is a dataset of 32x32 images in 10 categories, collected by Alex
Krizhevsky, Vinod Nair, and Geoffrey Hinton. It is often used to evaluate
machine learning algorithms. You can download this dataset from https://
www.cs.toronto.edu/∼kriz/cifar.html.

**(a)** For each category, compute the mean image and the first 20 principal
components. Plot the error resulting from representing the images of each
category using the first 20 principal components against the category.

**(b)** Compute the distances between mean images for each pair of classes. Use
principal coordinate analysis to make a 2D map of the means of each
categories. For this exercise, compute distances by thinking of the images
as vectors.

**(c)** Here is another measure of the similarity of two classes. For class A and
class B, define E(A → B) to be the average error obtained by representing
all the images of class A using the mean of class A and the first 20 principal
components of class B. Now define the similarity between classes to be
(1/2)(E(A → B)+E(B → A)). Use principal coordinate analysis to make
a 2D map of the classes. Compare this map to the map in the previous
exercise – are they different? why?
