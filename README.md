# Credit_Risk_Analysis

## Overview

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we are tasked to employ different techniques to train and evaluate models with unbalanced classes. We used imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we oversampled the data using the RandomOverSampler and SMOTE algorithms, and undersampled the data using the ClusterCentroids algorithm. Then, we used a combinatorial approach of over and undersampling using the SMOTEENN algorithm. Next, we compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. 


## Results

We observed the results of all six machine learning models based on three different criteria:

* Accuracy Score - a measure of how likely a model is to label all predictions correctly.
* Preciscion - a classifier's ability to accurately label samples and minimize false positives or negatives
* Recall (Sensitivity) - a classifier's ability to find all the positive or negative samples. In this scenario, the higher the recall, the less chance there is that a high risk applicant will be classified as low risk and vice versa.

#### Naive Random Oversampling
The first model was trained with data sampled using the naive random overampling technique. In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. Oversampling addresses class imbalance by duplicating or mimicking existing data.
* Balanced Accuracy Score: This model accurately predicts credit risk 64% of the time when the minority class is balanced by oversampling.
* Preciscion: The precision of this model is 0.01 for high risk and 1.00 for low risk applicants. This means that 100% of the predicted low risk      
  applicants are actually low risk, but only 1% of the predicted high risk applicants are actually high risk.
* Recall: The recall of this model is 0.66 for high risk, and 0.62 for low risk applicants. 

Results shown from our code:

![Screen Shot 2022-06-13 at 10 56 25 AM](https://user-images.githubusercontent.com/98566486/173382710-65fc307f-73b1-4904-a69e-d5a8c821ea64.png)

#### SMOTE Oversampling
The synthetic minority oversampling technique (SMOTE) is another oversampling approach to deal with unbalanced datasets. In SMOTE, like random oversampling, the size of the minority is increased. The key difference between the two lies in how the minority class is increased in size. In SMOTE, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created. It's important to note that although SMOTE reduces the risk of oversampling, it does not always outperform random oversampling. Another deficiency of SMOTE is its vulnerability to outliers.

* Balanced Accuracy Score: The SMOTE oversampling model has a slightly higher accuracy score than Naive ROS. This model makes accurate predictions of credit risk 65% of the time.
* Preciscion: SMOTE oversampling gives the same model preciscion score as the model trained with Naive ROS (1.00 and 0.01 for low and high risk). Both models inaccurately classify 90% of high risk applicants as low risk.
* Recall: The recall for this model puts 61% of high risk applicants are categorized as high risk and 69% of low risk applicants are classified as low risk.

Results shown from our code:

![Screen Shot 2022-06-13 at 11 03 15 AM](https://user-images.githubusercontent.com/98566486/173384162-1290e986-0cb8-40aa-92bc-59d27cb4a1f1.png

#### Cluster Centroids (Undersampling)

Undersampling is another technique to address class imbalance. Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased. Undersampling only uses actual data. On the other hand, undersampling involves loss of data from the majority class. Furthermore, undersampling is practical only when there is enough data in the training set. There must be enough usable data in the undersampled majority class for a model to be useful.

Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.

* Balanced Accuracy Score: Undersampling the majority class gives the accuracy score of 65%, which is almost same as the SMOTE oversampling score.
* Preciscion: The precision scores for this model are the same as the first two models (1.00 and 0.01 for low and high risk).
* Recall: THe recall scores for this model are also the lowest thus far with overall average of 40% for all applicants. 

Results shown from our code:




Summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
