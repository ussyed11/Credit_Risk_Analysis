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

![Screen Shot 2022-06-13 at 11 03 15 AM](https://user-images.githubusercontent.com/98566486/173384162-1290e986-0cb8-40aa-92bc-59d27cb4a1f1.png)

#### Cluster Centroids (Undersampling)

Undersampling is another technique to address class imbalance. Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased. Undersampling only uses actual data. On the other hand, undersampling involves loss of data from the majority class. Furthermore, undersampling is practical only when there is enough data in the training set. There must be enough usable data in the undersampled majority class for a model to be useful.

Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.

* Balanced Accuracy Score: Undersampling the majority class gives the accuracy score of 65%, which is almost same as the SMOTE oversampling score.
* Preciscion: The precision scores for this model are the same as the first two models (1.00 and 0.01 for low and high risk).
* Recall: THe recall scores for this model are also the lowest thus far with overall average of 40% for all applicants. 

Results shown from our code:

![Screen Shot 2022-06-13 at 11 33 05 AM](https://user-images.githubusercontent.com/98566486/173390437-571959e0-d79e-48d3-b559-fc916cab72b1.png)

#### SMOTEENN (Combination Sampling)

We used SMOTEENN technique in this model. As with SMOTE, the minority class is oversampled; however, an undersampling step is added, removing some of each class's outliers from the dataset. The result is that the two classes are separated more cleanly. Resampling with SMOTEENN did not work miracles, but some of the metrics show an improvement over undersampling.

* Balanced Accuracy Score: This model accurately predicts credit risk 64% of the time when the classes are balanced by combination over and undersampling.
* Preciscion: The precision scores for this model are the same as the first three models.
* Recall: This model correctly classifies 72% of high risk applicants and 57% of low risk applicants. This model has the best sensitivity for detecting high risk applicants out of all four sampling models.

Results shown from our code:

![Screen Shot 2022-06-13 at 11 38 01 AM](https://user-images.githubusercontent.com/98566486/173391398-62e39834-49ef-4663-a67b-f1917985a96e.png)

#### Balanced Random Forest Classifier
We next tried two ensembles models, which improves overall model performance by combining multiple models to help improve accuracty and decrease variance. The Random Forests Classifier is composed of several small decision trees created from random sampling. By using the Balanced Random Forests, we oversample from the minority class to balance the classes.

Balanced Accuracy Score: This model accurately predicts credit risk 66% of the time when multiple models are combined and the minority class is balanced by oversampling.
Preciscion: This model has the highest precision for classifying high risk applicants compared to models built from sampling techniques alone, but with a precision score of 0.72 for high risk and 1.0 for low risk applicants. This model has the same preciscion score for classifying low risk applicants as the previous models (100%).
Recall: This model correctly identifies 100% of low risk applicants as low risk, and 32% of high risk applicants as high risk. The high recall score for low risk makes this ensemble model a better performer than models built from sampling techniques alone.

Our findings:

![Screen Shot 2022-06-13 at 11 46 46 AM](https://user-images.githubusercontent.com/98566486/173393060-26d6fe7e-9320-4223-b90b-4160e71fe9c4.png)

#### Easy Ensemble AdaBoost Classifier
The final model was built using an easy ensemble classifier with adaptive boosting, or AdaBoost. Using AdaBoost, a model is trained and then evaluated. The errors of the first model are given extra weight when the subsequent model is trained and so on until the error rate is minimized.

* Balanced Accuracy Score: This model accurately predicts credit risk 92.5% of the time when multiple models are trained sequentially on a balanced dataset to minimize error.
* Preciscion: The precision score for correclty identifying high risk applicants is 7%, which is the highest for all 6 models. The preciscion score for low risk applicants is 100%, which is the same as the other models.
* Recall: In this model, 91% of high risk and 94% of low risk applicants were correctly identified, which is the highest recall score of all the models.

Our results are shown below:

![Screen Shot 2022-06-13 at 11 51 08 AM](https://user-images.githubusercontent.com/98566486/173393854-40438223-d280-4aac-96ca-ccd8c32eea54.png)

## Summary
Based on the analysis, it was clear that none of the models used here would have a high enough accuracy score to be used for credit risk evaluation since there was only little improvement in accuracy and the overall performances were too unsatisfactory to be reputably used in loan lending decision making. Out of all six models, the EasyEnsembleClassifer model yielded the best results with an accuracy rate of 92.5% and a 7% precision rate when predicting "High Risk candidates. The sensitivity rate (aka recall) was also the highest at 91% compared to the other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 94% and an F1 score of 97%. Therefore, if a model needed to be recommended to perform this type of analysis, then the EasyEnsembleClassifer model would be the clear choice.
