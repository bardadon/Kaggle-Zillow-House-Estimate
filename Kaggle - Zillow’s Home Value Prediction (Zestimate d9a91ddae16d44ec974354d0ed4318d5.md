# Kaggle -  Zillow’s Home Value Prediction (Zestimate)

Category: Data Science Projects
Created: Mar 11, 2021
Description: Can you improve the algorithm that changed the world of real estate?
Featured: No

- **What should be in the presentation?**
    1. A clear explanation of the problem and why its important.
    2. A summary of the data cleaning and exploration, including visualizations
    3. How I created a baseline
    4. My logic for selecting models to test, tuning models that I selected, and measuring the efficacy, e.g. why did I choose to measure the model using RMSE over R^2?
    5. The results I got for each model, also include a visualization.
    6. My approach to training the final model and making predictions.
    7. A summary of the project: results, findings, and areas that could be improved or explored in the future.

# Zillow House Estimate Project

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
 	Id 	MSSubClass 	MSZoning 	LotFrontage 	LotArea 	Street 	Alley 	LotShape 	LandContour 	Utilities 	... 	PoolArea 	PoolQC 	Fence 	MiscFeature 	MiscVal 	MoSold 	YrSold 	SaleType 	SaleCondition 	SalePrice
1181 	1182 	120 	RM 	64.0 	5587 	Pave 	NaN 	IR1 	HLS 	AllPub 	... 	0 	NaN 	NaN 	NaN 	0 	11 	2008 	New 	Partial 	392500
261 	262 	60 	RL 	69.0 	9588 	Pave 	NaN 	IR1 	Lvl 	AllPub 	... 	0 	NaN 	NaN 	NaN 	0 	11 	2007 	New 	Partial 	276000
1113 	1114 	20 	RL 	66.0 	8923 	Pave 	NaN 	Reg 	Lvl 	AllPub 	... 	0 	NaN 	NaN 	NaN 	0 	5 	2007 	WD 	Normal 	134500
1389 	1390 	50 	RM 	60.0 	6000 	Pave 	NaN 	Reg 	Lvl 	AllPub 	... 	0 	NaN 	NaN 	NaN 	0 	3 	2007 	WD 	Normal 	131000
969 	970 	190 	RL 	75.0 	10382 	Pave 	NaN 	Reg 	Lvl 	AllPub 	... 	0 	NaN 	NaN 	NaN 	0 	3 	2006 	ConLD 	Normal 	140000

5 rows × 81 columns
