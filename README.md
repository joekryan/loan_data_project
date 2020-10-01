# Loan Data Project

For this project I'm using a kaggle dataset from [Lending Club, a US lender](https://www.kaggle.com/wendykan/lending-club-loan-data). I will use this dataset to create a classifier to predict the loan status (fully paid or charged off) and then use Boruta feature selection to select the most important features and reduce the model's complexity.

## Methods Used
- Pandas
- Oversampling/Undersampling
- SMOTE
- ADASYN
- Logistic Regression
- GridSearchCV
- Boruta Feature Selection. 

## Initial EDA

<p align="left">
  <img src="https://github.com/joekryan/loan_data_project/blob/master/images/loan_status.png">
</p>

Once the data has been cleaned, their is a class imbalance of approximately 80:20. This is marginally imblanaced, but not to the extreme extent of, for example, classifying fraudulent credit card transactions, where the minority class (fraudulent transactions) would consist of much less than 1% of the dataset. Most machine learning models have a frequency bias, which means that they will be better at classifying the class that occurs more often, as there is more training data. However, if you have a dataset that is only marginally imbalanced, say 20:80 rather than 1:99, is it still necessary to be wary of imbalance?

To test this, I decided to do a comparison of 5 different methods of dealing with this imbalance:

- Do nothing (i.e. use the imbalanced dataset)
- Oversampling (duplicate minority class samples)
- Undersampling (randomly select a subset of majority class samples so both classes are equal)
- SMOTE (Synthetic Minority Oversampling TEchnique - creates new samples by selecting examples that are close in the feature space, drawing a line between them and creating a new sample at a point along that line)
- ADASYN (ADAptive SYNthetic sampling - similar to SMOTE, but uses a weighted distribution to generate new samples, generating more synthetic data for examples that are harder to learn)

## Metric

First, however, I needed to decide which metric to use. Accuracy is not a good metric to use for imbalanced data sets because if a dataset is very imbalanced then a model can be highly accurate just by always predicting the majority class. I.e. if a target class is 99% True and 1% False, a classifier can be 99% accurate just by always predicting True. Clearly this is not ideal

Instead, I will use recall as the metric. Recall is the sum of true positives divided by the sum of the true positives PLUS the false negatives. I.e., of all the true positives that exist, how many did your model identify? A positive in this case being the minority class, a loan that is charged off.

<p align="left">
  <img src="https://miro.medium.com/max/700/1*1f0Rw_N_1Dp3aZwPyGUNpA.png">
</p>

## Imbalance comparison

<p align="left">
  <img src="https://github.com/joekryan/loan_data_project/blob/master/images/imbalance.png">
</p>

To compare the different models, I used a logistic regression model, with a different pipeline (optimising hyperparameters with GridSearchCV) for each of the 5 methods. As can be seen, all of the methods scored highly, with SMOTE scoring the highest. However, the tiny difference in recall score between SMOTE/ADASYN (0.994) and Oversampling (0.993) barely seems worth the extra time and computational power that is required for the former techniques. In fact even the base model, unadjusted for imbalance, had a recall score of 0.969.





