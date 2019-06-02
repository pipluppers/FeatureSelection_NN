# Feature Selection using Nearest Neighbor
## Thanks to Eamonn Keogh for overseeing this project!

Input Files:

1. Big dataset with a class label and 50 features (numerical values)

2. Small dataset with a class label and 10 features (numerical values)

3. Python program file



The k-Nearest Neighbor classifier assigns a label to a data point based on the labels of the k closest neighboring points. This model is accurate but not very fast, and it is also sensitive to irrelevant features.
Because of its sensitivity, we can use the classifier to find the best set of features of a dataset. We will know that an irrelevant feature will have a noticeable effect on our labels. This fact will be used to derive the best subset.
Three algorithms will be explored here. The first is forward selection, which adds features one-by-one and picks one based on its accuracy (here we use the nearest neighbor (or 1-nearest neighbor) method to measure the accuracy.
Cross validation, specifically leave-one-out, is used to train our model. The leave-one-out method selects one data instance as the test set and the rest as the training set. This results in good accuracy because the training set is still roughly the size of the original set. However, this method is prone to long runtimes, especially when the dataset becomes large.
