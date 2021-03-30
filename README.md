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

In order to estimate Zillow's houses, in this competition, we were given a dataset of 2919 records of houses where each house is defined by 81 features. 

These features are mostly indications of wealth and house quality, such as: having a pool or not, the square-foot area of the street around the house, overall condition, year built, and so on.


## Sample Data
|   |  Id | MSSubClass | MSZoning | LotFrontage | LotArea | Street | Alley | LotShape | LandContour | Utilities | ... | PoolArea | PoolQC | Fence | MiscFeature | MiscVal | MoSold | YrSold | SaleType | SaleCondition | SalePrice |
|--:|----:|-----------:|---------:|------------:|--------:|-------:|------:|---------:|------------:|----------:|----:|---------:|-------:|------:|------------:|--------:|-------:|-------:|---------:|--------------:|----------:|
|   | 995 | 20         | RL       | 96.0        | 12456   | Pave   | NaN   | Reg      | Lvl         | AllPub    | ... | 0        | NaN    | NaN   | NaN         | 0       | 7      | 2009   | WD       | Normal        | 337500    |
|   | 885 | 20         | RL       | 65.0        | 7150    | Pave   | NaN   | Reg      | Lvl         | AllPub    | ... | 0        | NaN    | GdWo  | NaN         | 0       | 7      | 2009   | WD       | Normal        | 100000    |
|   | 60  | 20         | RL       | 60.0        | 7200    | Pave   | NaN   | Reg      | Bnk         | AllPub    | ... | 0        | NaN    | MnPrv | NaN         | 0       | 1      | 2008   | WD       | Normal        | 124900    |
|   | 537 | 60         | RL       | 57.0        | 8924    | Pave   | NaN   | IR1      | Lvl         | AllPub    | ... | 0        | NaN    | NaN   | NaN         | 0       | 7      | 2008   | WD       | Normal        | 188000    |
|   | 78  | 50         | RM       | 50.0        | 8635    | Pave   | NaN   | Reg      | Lvl         | AllPub    | ... | 0        | NaN    | MnPrv | NaN         | 0       | 1      | 2008   | WD       | Normal        | 127000    |





__Created__: Mar 11, 2021

__Author__: Bar Dadon

__Email__: bdadon50@gmail.com

