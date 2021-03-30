# Kaggle Competition -  Zillow’s Home Value Prediction (Zestimate)

Zillow is a leading real estate and rental marketplace dedicated to empowering consumers with data, they serve the full lifecycle of owning and living in a home: buying, selling, renting, financing, remodeling, and more.

Zillow estimates the value of their homes and calls it "Zestimate". 

*From Zillow*:

> *" “Zestimates” are estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property."*

In 5/24/2017 Zillow has initiated a Kaggle competition in order to improve their Zestimate model.

This is my take on their competition.

# The Problem

How much is a house really worth? answering this question can be a difficult task for ordinary people trying to purchase their first house.

A house is worth whatever people will pay for it and it depends on the market, the quality of the house, the location, and so on.

Unfortunately, it also depends on who you're asking, whether you’re asking a lender, an agent, or a county tax assessor.

In other words, knowing how to calculate and estimate a home's value independently can have huge benefits to consumers who are trying to get by and buy a house. It can bring *control* and *power* in negotiations if consumers would know how much the house that they are interested in is actually worth.

# The Data

In order to estimate Zillow's houses, in this competition, we were given a data-set of 2919 records of houses where each house is defined by 80 features plus a target variable which indicates the Sale Price at which the house was sold. 

The features are mostly indications of wealth and house quality, such as: having a pool or not, the square-foot area of the street around the house, overall condition, year built, and so on.


## Sample Data
|   |  Id | MSSubClass | MSZoning | LotFrontage | LotArea | Street | Alley | LotShape | LandContour | Utilities | ... | PoolArea | PoolQC | Fence | MiscFeature | MiscVal | MoSold | YrSold | SaleType | SaleCondition | SalePrice |
|--:|----:|-----------:|---------:|------------:|--------:|-------:|------:|---------:|------------:|----------:|----:|---------:|-------:|------:|------------:|--------:|-------:|-------:|---------:|--------------:|----------:|
|   | 995 | 20         | RL       | 96.0        | 12456   | Pave   | NaN   | Reg      | Lvl         | AllPub    | ... | 0        | NaN    | NaN   | NaN         | 0       | 7      | 2009   | WD       | Normal        | 337500    |
|   | 885 | 20         | RL       | 65.0        | 7150    | Pave   | NaN   | Reg      | Lvl         | AllPub    | ... | 0        | NaN    | GdWo  | NaN         | 0       | 7      | 2009   | WD       | Normal        | 100000    |
|   | 60  | 20         | RL       | 60.0        | 7200    | Pave   | NaN   | Reg      | Bnk         | AllPub    | ... | 0        | NaN    | MnPrv | NaN         | 0       | 1      | 2008   | WD       | Normal        | 124900    |
|   | 537 | 60         | RL       | 57.0        | 8924    | Pave   | NaN   | IR1      | Lvl         | AllPub    | ... | 0        | NaN    | NaN   | NaN         | 0       | 7      | 2008   | WD       | Normal        | 188000    |
|   | 78  | 50         | RM       | 50.0        | 8635    | Pave   | NaN   | Reg      | Lvl         | AllPub    | ... | 0        | NaN    | MnPrv | NaN         | 0       | 1      | 2008   | WD       | Normal        | 127000    |

## Feature Selection

Since the data has a lot of features compared to the amount of observations(records of houses), It was necessary to select only the best predictors in order to avoid creating a slow and highly biased model.

I've divided the features in the data set to numerical and categorical, each group have gone through a different set of feature selection tests.

### Numerical Features

The N*umerical* feature selection tests were:

1. Variance Threshold -
    - Finding features with a variance lower than a certain threshold.
    - Threshold is set to 0.
2. Correlation with other features -
    - Finding features with *High* correlation with other features.
    - High correlation is defined as higher than 0.85 or lower than -0.85.
3. Correlation with the target variable - 
    - Finding features with *Low* correlation with the target variable.
    - Low correlation is defined as between [0.05, -0.05].

### Categorical Features:

The C*ategorical* feature selection test were:

1. Chi-Square test -
    - Null Hypothesis- The two *Categorical* variables are independent.
    - Alternate Hypothesis- The two *Categorical* variables are dependent.

### Feature Selection Summary

Out of a total of 80 features, 36 did NOT meet any one of the tests for feature selection and therefore were dropped.

```python
overall_features_to_drop = features_dropped_var.union(features_dropped_const, features_to_drop_pairs, features_to_drop_target,features_to_drop_chi )
print('The features that did not meet any one of the feature selection tests are: \n{}'.format(overall_features_to_drop))
```

`The features that did not meet any one of the feature selection tests are: 
{'Fence', 'GarageYrBlt', 'BsmtFinType1', 'GarageCond', 'BsmtFinType2', 'PoolQC', 'BsmtFinSF2', 'Exterior2nd', 'MiscFeature', 'LandContour', 'HeatingQC', 'GarageType', 'RoofStyle', 'KitchenAbvGr', 'HouseStyle', 'Condition2', 'Alley', 'Condition1', 'PoolArea', 'Functional', 'PavedDrive', 'GarageCars', 'LowQualFinSF', 'MoSold', '1stFlrSF', 'RoofMatl', 'Electrical', 'Exterior1st', 'TotRmsAbvGrd', 'LandSlope', 'Utilities', 'BsmtHalfBath', 'BldgType', '3SsnPorch', 'YrSold', 'MiscVal'}`

## Building the Model

The model was built using the ensemble method called Gradient Boosting.

Gradient Boosting can handle data sets with a relatively high number of features compared to observations, also since Boosting methods rely on weak learners to learn from the errors of previous learners(In this case Decision trees) it is suitable in cases where there are no extremely important feature.

### Tuning Model

Hyper-parameter tuning is especially significant for Gradient Boosting models since they are prone to overfitting.


__Created__: Mar 11, 2021

__Author__: Bar Dadon

__Email__: bdadon50@gmail.com

