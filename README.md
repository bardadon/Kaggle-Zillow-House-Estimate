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

```python
df.sample(5)
```






__Created__: Mar 11, 2021

__author__ = Bar Dadon

__email__ = bdadon50@gmail.com

