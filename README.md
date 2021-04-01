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

1. <ins>Variance Threshold</ins> 
    - Finding features with a variance lower or equal to a certain threshold.
    - Threshold is set to 0.
2. <ins>Correlation with other features</ins> 
    - Finding features with *High* correlation with other features.
    - High correlation is defined as higher than 0.85 or lower than -0.85.
3. <ins>Correlation with the target variable</ins>  
    - Finding features with *Low* correlation with the target variable.
    - Low correlation is defined as between [0.05, -0.05].

### Categorical Features:

The C*ategorical* feature selection test were:

1. <ins>Chi-Square test</ins> 
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

Gradient Boosting can handle data sets with a relatively high number of features compared to observations, also since Boosting methods rely on weak learners to learn from the errors of previous learners(In this case Decision trees) it is suitable in cases where there are no extremely important features.

### Tuning Model

Hyper-parameter tuning is especially significant for Gradient Boosting models since they are prone to overfitting.
First, In order to understand the model's hyper-parameters better I plotted learning curves. These learning curves can help learn about the model's goodness of fit as a function of any hyper-parameter.

Since Gradient Boosting is prone to overfitting it was important to tune hyper-parameters that would act as regularization for the model.

The hyper-parameters I tuned are: Max Depth, Max Features, and Learning Rate. 

1. <ins>__Max Depth__</ins>
![index1111](https://user-images.githubusercontent.com/65648983/113309255-a3648b00-930f-11eb-99de-76cfcbae6ba2.png)

| max_depth |  train_error | validation_error |                       | Min Validation MAE |Optimal Max Depth| 
|----------:|-------------:|-----------------:|                       |-------------------:|-----------------:|
| 1         | 18090.849971 | 20221.128531     |                        |15843.219232272917  |  4              |
| 2         | 13381.785825 | 16609.755916     |
| 3         | 10423.167237 | 16066.300709     |
| 4         | 7587.704458  | 15843.219232     |
| 5         | 5072.825906  | 16624.464742     |
| 6         | 2819.594728  | 16781.484089     |
| 7         | 1461.937664  | 18296.192063     |
| 8         | 643.510065   | 18995.070019     |
| 10        | 61.143895    | 21707.195643     |
| 15        | 1.575364     | 23671.361496     |
| 20        | 1.534816     | 24709.851204     |

| Min Validation MAE |Optimal Max Depth| 
|-------------------:|---------------------:|
|15843.219232272917  |  4 



2. <ins>__Max Features__</ins>

![index1231](https://user-images.githubusercontent.com/65648983/113309959-5fbe5100-9310-11eb-9b97-2bc1f6d64624.png)
|max_features |  train_error     |validation_error
|------------:|-----------------:|--------------:|
| 1.0         | 21967.308221     | 25737.532621 |
| 5.0         | 14363.436803     | 17989.454864 |
| 10.0        | 12705.556069     | 16963.207451 |
| 20.0        | 11672.629323     | 15917.558761 |
| 40.0        | 10972.956964     | 15348.508493 |
| 60.0        | 10363.054822     | 15527.005733 |
| 80.0        | 10563.508034     | 15742.870212 |
| NaN         | 10423.167237     | 16038.734329 |   


3. <ins>__Learning Rate__</ins>

![index333](https://user-images.githubusercontent.com/65648983/113310382-d9eed580-9310-11eb-8688-ab353f0532f3.png)
|learning_rate| train_error     | validation_error|              
|------------:|-----------------:|--------------:|
| 0.001       | 53597.429015     | 53631.299545 |
| 0.003       | 46667.218318     | 46849.248860 |
| 0.005       | 40937.877787     | 41211.304561 |
| 0.010       | 30513.193567     | 31102.857673 |
| 0.030       | 15701.328920     | 19082.082642 |
| 0.050       | 12649.763485     | 16846.771170 |
| 0.100       | 10423.167237     | 16112.084974 |
| 0.300       | 6750.457311      | 16112.665356 |
| 0.500       | 4793.009255      | 17226.237856 |

__Created__: Mar 11, 2021

__Author__: Bar Dadon

__Email__: bdadon50@gmail.com

