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

## Feature Selection with Boruta

<p align="left">
  <img src="https://github.com/joekryan/loan_data_project/blob/master/images/features.png">
</p>

In the data preparation process, as a result of a large dataset and encoding categorical features, the model ended up with almost 1600 features. This not only greatly increases the copmutational power needed to run predictive models, but also makes it more difficult to explain the model to a non-technical audience. 

A good data scientist will want to select the important features from the model and remove ones that are not interesting or relevant to your model. There are many methods of feature selection, but most of them in some way involve selecting an arbitrary threshold. This is where Boruta comes in. 

The idea behind Boruta is quite simple. First we duplicate all the columns, then the values within each column are shuffled, creating a twin 'shadow column' for each column. Then a classifier is trained on this dataset (typically a Random Forest), that can generate a feature importance value for each feature and shadow feature. Now, we take the importance of each original features and compare it with a threshold. This time, the threshold is defined as the highest feature importance recorded among the shadow features. When the importance of a feature is higher than this threshold, this is called a "hit". The idea is that a feature is useful only if it's capable of doing better than the best randomized feature.


<p align="left">
  <img src="https://cdn-images-1.medium.com/max/800/1*xYjfAdGeoOOQNVkPvZvbvw.png">
</p>

But, as is often the case with machine learning, just one iteration is never enough. 20 trials are more reliable than 1 trial and 100 trials are more reliable than 20 trials. At every iteration we check if a given feature is doing better then expected than random chance. We do this by simply comparing the number of times a feature did better than the shadow features using a binomial distribution. Let's take a feature and say we have absolutely no clue if it's useful or not. What is the probability that we shall keep it? The maximum level of uncertainty about the feature is expressed by a probability of 50%, like tossing a coin. Since each independent experiment can give a binary outcome (hit or no hit), a series of n trials follows a binomial distribution.


<p align="left">
  <img src="https://cdn-images-1.medium.com/max/800/1*XMlUyvnqFwaQA8EwFdUnOw.png">
</p>

In Boruta, there is not a hard threshold between a refusal and an acceptance area. Instead, there are 3 areas:

* an area of refusal (the red area): the features that end up here are considered as noise, so they are dropped;
* an area of irresolution (the blue area): Boruta is indecisive about the features that are in this area;
* an area of acceptance (the green area): the features that are here are considered as predictive, so they are kept.

<p align="left">
  <img src="https://github.com/joekryan/loan_data_project/blob/master/images/boruta_forest.png">
</p>

My boruta implementation in this case used a Random Forest classifier. The only change from the default that I made was to set the percentage ('perc') to 90. This means that the threshold is set to 90% of the highest shadow value, rather than its full value. This means there will be a trade off where more false positives will be picked as relevant but also the less relevant features will be left out.

<p align="left">
  <img src="hhttps://github.com/joekryan/loan_data_project/blob/master/images/boruta_features.png">
</p>

After this first implementation, 35 features were selected, a decrease in feature number by almost 98%! And recall was still 99%!

As you can see, some of these features might not pass a 'common sense' test for someone who is not familiar with machine learning and dummy variables. Plus, thinking about the business problem, having several predictive features as specific months when credit was issued will not be useful going forwards. So removing these gives a final group of 18 features!

<p align="left">
  <img src="hhttps://github.com/joekryan/loan_data_project/blob/master/images/features.png">
</p>

Running these through a logistic regression still provides a predictive model with a recall of 99%! Additionally, all of these features are things that can be easily explained and understood to a non-technical audience!

## Conclusion

* Computationally expensive techniques for dealing with imbalances such as SMOTE/ADASYN may not be worth it when the dataset is only marginally imbalanced.
* Feature Selection can greatly decrease the complexity of the model, saving computational time/power and making it easier to explain the model to a non-technical audience

