# Feature Selection using Nearest Neighbor
## Thanks to Eamonn Keogh for overseeing this project!

Input Files:

1. Big dataset with a class label and 50 features (numerical values)

2. Small dataset with a class label and 10 features (numerical values)

3. Python3 program file

Real world data contains multitudes of irrelevant entires and features. A modern problem is to derive the useful features that constitute a label given a dataset and ignore the rest. An example of labels could be edible food versus non-edible foods and the relevant features could include chemicals and origin while non-relevant features may include temperature. Both of these two above sample input datasets contain multiple irrelevant features. The goal of this project is to filter these features out and determine which fields are actually significant in determining the correct label.

The classifier used here is k-Nearest Neighbor with k being equal to 1. This model is accurate but is not very fast, and is sensitive to irrelevant features. Because of this sensitivity, we can use it to our advantage to find relevant features. Since each feature is composed of numerical values, each row instance can be represented as a multidimensional point. In order to find the closest neighbor, the Euclidean distance is computed between the desired point and all of the others. Of course, the point that yields the lowest Euclidean distance is the closest. From there, we say our desired row has the label of the closest row.

Cross validation (specifically leave-one-out, which is the idea of having the test set be one row and the training set be the remaining rows) is used to train the model. This results in good accuracy but the runtime can be horrendously long if the dataset contains a large amount of rows. Three algorithms will be explored here: forward selection, backwards elimination, and forward selection with pruning.

Forward selection is an algorithm that starts with an empty initial set. Features will be considered iteratively one-by-one and the feature that yielded the highest accuracy will be added to the set. The process is as follows:

- Initially consider one feature. Use this feature only when performing the nearest neighbor classifier with the test and training set provided through leave-one-out cross validation.

- Once we have done this classifier with every row, count the number correct and divide by total. This is the accuracy if we decided to add this feature.

- Then consider the next feature and repeat. If this feature yields a higher accuracy than the previous ones, ignore them for now. Repeat for remaining features.

- Once all features are considered, add the one with the highest accuracy to a list. This is the list of relevant features. Next pick another feature that is not in this new list. Repeat all the above steps with now considering this new one along with the known relevant features.

- Repeat until adding a new feature to the list decreases the overall accuracy. This final list is the output list of relevant features.

Backwards elimination follows the exact same concept except for instead of starting with an empty list of relevant features and adding to it, this method starts with the full list of features and subtracts from it. 

The pruning method is alpha-beta pruning. This improves the search runtime of forward selection by ending searches early if there is no way for the feature to be relevant. As each feature is searched using the nearest neighbor classifier, a tracking variable is used to count the number of incorrect labels. If the feature ever yielded one more wrong label than this variable, we can stop running cross validation and nearest neighbor with this feature because it is guaranteed to be worse than a previous one. This removes redundant calls to the classifier.

Accuracy of the output subset of relevant features was oversaw by the professor.
