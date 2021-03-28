```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import missingno as msno
import sweetviz as sv
```

# Loading Data


```python
# now read in the functions
from Helpers import read_dataframe
```


```python
df = read_dataframe(path ='train.csv', analyze = True )
```

    
    
    '[1m'
    ******************************
    Data Report
    ******************************
    Number of rows:: 1460
    Number of columns:: 81
    
    
    Column Names:: ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']
    
    
    Column Data Types::
     Id                 int64
    MSSubClass         int64
    MSZoning          object
    LotFrontage      float64
    LotArea            int64
                      ...   
    MoSold             int64
    YrSold             int64
    SaleType          object
    SaleCondition     object
    SalePrice          int64
    Length: 81, dtype: object
    
    
    Columns with Missing Values:: ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
    
    
    Number of rows with Missing Values:: 1460
    
    
    Sample Indices with missing data:: [0, 1, 2, 3, 4]
    
    
    General Stats::
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     1452 non-null   object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    None
    
    
    Summery Stats(Numerical): 
                    Id   MSSubClass  LotFrontage        LotArea  OverallQual  \
    count  1460.000000  1460.000000  1201.000000    1460.000000  1460.000000   
    mean    730.500000    56.897260    70.049958   10516.828082     6.099315   
    std     421.610009    42.300571    24.284752    9981.264932     1.382997   
    min       1.000000    20.000000    21.000000    1300.000000     1.000000   
    25%     365.750000    20.000000    59.000000    7553.500000     5.000000   
    50%     730.500000    50.000000    69.000000    9478.500000     6.000000   
    75%    1095.250000    70.000000    80.000000   11601.500000     7.000000   
    max    1460.000000   190.000000   313.000000  215245.000000    10.000000   
    
           OverallCond    YearBuilt  YearRemodAdd   MasVnrArea   BsmtFinSF1  ...  \
    count  1460.000000  1460.000000   1460.000000  1452.000000  1460.000000  ...   
    mean      5.575342  1971.267808   1984.865753   103.685262   443.639726  ...   
    std       1.112799    30.202904     20.645407   181.066207   456.098091  ...   
    min       1.000000  1872.000000   1950.000000     0.000000     0.000000  ...   
    25%       5.000000  1954.000000   1967.000000     0.000000     0.000000  ...   
    50%       5.000000  1973.000000   1994.000000     0.000000   383.500000  ...   
    75%       6.000000  2000.000000   2004.000000   166.000000   712.250000  ...   
    max       9.000000  2010.000000   2010.000000  1600.000000  5644.000000  ...   
    
            WoodDeckSF  OpenPorchSF  EnclosedPorch    3SsnPorch  ScreenPorch  \
    count  1460.000000  1460.000000    1460.000000  1460.000000  1460.000000   
    mean     94.244521    46.660274      21.954110     3.409589    15.060959   
    std     125.338794    66.256028      61.119149    29.317331    55.757415   
    min       0.000000     0.000000       0.000000     0.000000     0.000000   
    25%       0.000000     0.000000       0.000000     0.000000     0.000000   
    50%       0.000000    25.000000       0.000000     0.000000     0.000000   
    75%     168.000000    68.000000       0.000000     0.000000     0.000000   
    max     857.000000   547.000000     552.000000   508.000000   480.000000   
    
              PoolArea       MiscVal       MoSold       YrSold      SalePrice  
    count  1460.000000   1460.000000  1460.000000  1460.000000    1460.000000  
    mean      2.758904     43.489041     6.321918  2007.815753  180921.195890  
    std      40.177307    496.123024     2.703626     1.328095   79442.502883  
    min       0.000000      0.000000     1.000000  2006.000000   34900.000000  
    25%       0.000000      0.000000     5.000000  2007.000000  129975.000000  
    50%       0.000000      0.000000     6.000000  2008.000000  163000.000000  
    75%       0.000000      0.000000     8.000000  2009.000000  214000.000000  
    max     738.000000  15500.000000    12.000000  2010.000000  755000.000000  
    
    [8 rows x 38 columns]
    
    
    Summery Stats(Objects): 
                  count unique     top  freq
    Id             1460   1460    1460     1
    MSSubClass     1460     15      20   536
    MSZoning       1460      5      RL  1151
    LotFrontage    1201    110      60   143
    LotArea        1460   1073    7200    25
    ...             ...    ...     ...   ...
    MoSold         1460     12       6   253
    YrSold         1460      5    2009   338
    SaleType       1460      9      WD  1267
    SaleCondition  1460      6  Normal  1198
    SalePrice      1460    663  140000    20
    
    [81 rows x 4 columns]
    
    
    Dataframe Sample Rows::
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>


    Visualizing Missing Values...
    


    
![png](output_3_3.png)
    


# Data Exploration 


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>994</th>
      <td>995</td>
      <td>20</td>
      <td>RL</td>
      <td>96.0</td>
      <td>12456</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>337500</td>
    </tr>
    <tr>
      <th>884</th>
      <td>885</td>
      <td>20</td>
      <td>RL</td>
      <td>65.0</td>
      <td>7150</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdWo</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>100000</td>
    </tr>
    <tr>
      <th>59</th>
      <td>60</td>
      <td>20</td>
      <td>RL</td>
      <td>60.0</td>
      <td>7200</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Bnk</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>124900</td>
    </tr>
    <tr>
      <th>536</th>
      <td>537</td>
      <td>60</td>
      <td>RL</td>
      <td>57.0</td>
      <td>8924</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>188000</td>
    </tr>
    <tr>
      <th>77</th>
      <td>78</td>
      <td>50</td>
      <td>RM</td>
      <td>50.0</td>
      <td>8635</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>127000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



## EDA1 - Duplicates


```python
# Lets check how many record have the same Id number
amount_of_duplicated_Id = df['Id'].duplicated().sum()
print("There are {} Duplicated ID's in the DataFrame".format(amount_of_duplicated_Id))
```

    There are 0 Duplicated ID's in the DataFrame
    


```python
df.duplicated().any()
```




    False



## EDA2 - Null Values


```python
# Checking how many records have missing values in each feature
df.isnull().sum().sort_values(ascending = False).head(20)
```




    PoolQC          1453
    MiscFeature     1406
    Alley           1369
    Fence           1179
    FireplaceQu      690
    LotFrontage      259
    GarageCond        81
    GarageType        81
    GarageYrBlt       81
    GarageFinish      81
    GarageQual        81
    BsmtExposure      38
    BsmtFinType2      38
    BsmtFinType1      37
    BsmtCond          37
    BsmtQual          37
    MasVnrArea         8
    MasVnrType         8
    Electrical         1
    Utilities          0
    dtype: int64



### Notes:
- The features with the highest amount of null values are all related to wealth(pool, fireplace, fence etc...)

## EDA3 - Unique Values


```python
# Importing functions from my Helper Module
from Helpers import extract_cat_features, extract_num_features
```


```python
# Extracting the categorical and numerical features from the DataFrame
cat_features = extract_cat_features(df)
num_features = extract_num_features(df)
```

### Unique Categorical Features


```python
# Checking how many unique values there are in each categorical feature
df[cat_features].nunique().sort_values(ascending=False).head(45)
```




    Neighborhood     25
    Exterior2nd      16
    Exterior1st      15
    Condition1        9
    SaleType          9
    Condition2        8
    HouseStyle        8
    RoofMatl          8
    Functional        7
    SaleCondition     6
    BsmtFinType2      6
    Heating           6
    Foundation        6
    GarageType        6
    RoofStyle         6
    BsmtFinType1      6
    ExterCond         5
    BldgType          5
    LotConfig         5
    MSZoning          5
    GarageCond        5
    GarageQual        5
    HeatingQC         5
    Electrical        5
    FireplaceQu       5
    LotShape          4
    LandContour       4
    MiscFeature       4
    Fence             4
    BsmtExposure      4
    BsmtCond          4
    KitchenQual       4
    MasVnrType        4
    ExterQual         4
    BsmtQual          4
    GarageFinish      3
    PavedDrive        3
    PoolQC            3
    LandSlope         3
    Utilities         2
    CentralAir        2
    Alley             2
    Street            2
    dtype: int64



#### Notes:

- There are no categories with high cardinallity 


### Unique Numerical Features


```python
# Checking how many unique values there are in each numerical feature
df[num_features].nunique().sort_values(ascending=False).head(40)
```




    Id               1460
    LotArea          1073
    GrLivArea         861
    BsmtUnfSF         780
    1stFlrSF          753
    TotalBsmtSF       721
    SalePrice         663
    BsmtFinSF1        637
    GarageArea        441
    2ndFlrSF          417
    MasVnrArea        327
    WoodDeckSF        274
    OpenPorchSF       202
    BsmtFinSF2        144
    EnclosedPorch     120
    YearBuilt         112
    LotFrontage       110
    GarageYrBlt        97
    ScreenPorch        76
    YearRemodAdd       61
    LowQualFinSF       24
    MiscVal            21
    3SsnPorch          20
    MSSubClass         15
    MoSold             12
    TotRmsAbvGrd       12
    OverallQual        10
    OverallCond         9
    PoolArea            8
    BedroomAbvGr        8
    YrSold              5
    GarageCars          5
    KitchenAbvGr        4
    Fireplaces          4
    BsmtFullBath        4
    FullBath            4
    HalfBath            3
    BsmtHalfBath        3
    dtype: int64



## EDA4 - Target Variable


```python
sns.histplot(df['SalePrice'], kde = True)
plt.xlabel('House Sale Price', size = 18)
plt.ylabel('Count', size = 18)
plt.title('Sale Price Distribution', size = 18)
plt.show()
```


    
![png](output_21_0.png)
    


### After Log Transformation


```python
sns.histplot(np.log(df['SalePrice']), kde = True)
plt.xlabel('House Sale Price', size = 18)
plt.ylabel('Count', size = 18)
plt.title('Sale Price Distribution', size = 18)
plt.show()
```


    
![png](output_23_0.png)
    


# Feature Selection - Numerical Features



```python
from Helpers import x_y_split, extract_num_features
```


```python
x, y = x_y_split(df)
```


```python
numerical_features = extract_num_features(x)
numerical_df = df.loc[:, numerical_features]
```


```python
numerical_df = numerical_df.drop('Id', axis = 1)
```


```python
numerical_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>0</td>
      <td>...</td>
      <td>548</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>0</td>
      <td>...</td>
      <td>460</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>0</td>
      <td>...</td>
      <td>608</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>0</td>
      <td>...</td>
      <td>642</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>0</td>
      <td>...</td>
      <td>836</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>



## Variance


```python
from sklearn.feature_selection import VarianceThreshold
from Helpers import variance_plot
```


```python
variance_plot(numerical_df)
```


    
![png](output_32_0.png)
    



```python
selector = VarianceThreshold(threshold=0)
selector.fit_transform(numerical_df)
```




    array([[6.000e+01, 6.500e+01, 8.450e+03, ..., 0.000e+00, 2.000e+00,
            2.008e+03],
           [2.000e+01, 8.000e+01, 9.600e+03, ..., 0.000e+00, 5.000e+00,
            2.007e+03],
           [6.000e+01, 6.800e+01, 1.125e+04, ..., 0.000e+00, 9.000e+00,
            2.008e+03],
           ...,
           [7.000e+01, 6.600e+01, 9.042e+03, ..., 2.500e+03, 5.000e+00,
            2.010e+03],
           [2.000e+01, 6.800e+01, 9.717e+03, ..., 0.000e+00, 4.000e+00,
            2.010e+03],
           [2.000e+01, 7.500e+01, 9.937e+03, ..., 0.000e+00, 6.000e+00,
            2.008e+03]])




```python
original_features_list = set(numerical_df.columns.tolist())
```


```python
features_to_keep = numerical_df.loc[:, selector.get_support()].columns.tolist()
features_to_keep = set(features_to_keep)
```


```python
features_dropped_var = original_features_list.difference(features_to_keep)
print('The features that did not meet the variance threshold criteria are: \n{}'.format(features_dropped_var))
```

    The features that did not meet the variance threshold criteria are: 
    set()
    

## Constant Features 


```python
from Helpers import constantish_features
```


```python
features_dropped_const = constantish_features(numerical_df, 0.95)
print('The features that did not meet the constant criteria are: \n{}'.format(features_dropped_const))
```

    The features that did not meet the constant criteria are: 
    ['LowQualFinSF', 'KitchenAbvGr', '3SsnPorch', 'PoolArea', 'MiscVal']
    

## Duplicate Features


```python
from Helpers import find_duplicated_columns
```


```python
find_duplicated_columns(df)
```

    No Duplicate Features
    

## Correlation

### Dropping highly correlated groups/pairs of features


```python
# now read in the functions
from Helpers import correlation_heatmap, get_pairs, top_abs_corrs, find_duplicated_columns, x_y_split
```


```python
correlation_heatmap(df)
```


    
![png](output_46_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YearBuilt</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>GrLivArea</th>
      <th>TotRmsAbvGrd</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>YearBuilt</th>
      <td>1.000000</td>
      <td>0.391452</td>
      <td>0.281986</td>
      <td>0.199010</td>
      <td>0.095589</td>
      <td>0.825667</td>
      <td>0.537850</td>
      <td>0.478954</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>0.391452</td>
      <td>1.000000</td>
      <td>0.819530</td>
      <td>0.454868</td>
      <td>0.285573</td>
      <td>0.322445</td>
      <td>0.434585</td>
      <td>0.486665</td>
    </tr>
    <tr>
      <th>1stFlrSF</th>
      <td>0.281986</td>
      <td>0.819530</td>
      <td>1.000000</td>
      <td>0.566024</td>
      <td>0.409516</td>
      <td>0.233449</td>
      <td>0.439317</td>
      <td>0.489782</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>0.199010</td>
      <td>0.454868</td>
      <td>0.566024</td>
      <td>1.000000</td>
      <td>0.825489</td>
      <td>0.231197</td>
      <td>0.467247</td>
      <td>0.468997</td>
    </tr>
    <tr>
      <th>TotRmsAbvGrd</th>
      <td>0.095589</td>
      <td>0.285573</td>
      <td>0.409516</td>
      <td>0.825489</td>
      <td>1.000000</td>
      <td>0.148112</td>
      <td>0.362289</td>
      <td>0.337822</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>0.825667</td>
      <td>0.322445</td>
      <td>0.233449</td>
      <td>0.231197</td>
      <td>0.148112</td>
      <td>1.000000</td>
      <td>0.588920</td>
      <td>0.564567</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>0.537850</td>
      <td>0.434585</td>
      <td>0.439317</td>
      <td>0.467247</td>
      <td>0.362289</td>
      <td>0.588920</td>
      <td>1.000000</td>
      <td>0.882475</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>0.478954</td>
      <td>0.486665</td>
      <td>0.489782</td>
      <td>0.468997</td>
      <td>0.337822</td>
      <td>0.564567</td>
      <td>0.882475</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Filtering the High Correlation pairs of features


```python
corr_df = top_abs_corrs(df)
corr_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>feature_2</th>
      <th>correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GarageCars</td>
      <td>GarageArea</td>
      <td>0.882475</td>
    </tr>
    <tr>
      <th>1</th>
      <td>YearBuilt</td>
      <td>GarageYrBlt</td>
      <td>0.825667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GrLivArea</td>
      <td>TotRmsAbvGrd</td>
      <td>0.825489</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TotalBsmtSF</td>
      <td>1stFlrSF</td>
      <td>0.819530</td>
    </tr>
  </tbody>
</table>
</div>



####  Ill use a RandomForest model to choose the most valuable features of each pair. 



```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from Helpers import who_to_drop
```


```python
features_to_drop_pairs = who_to_drop(corr_df, numerical_df, y)
print('The features to drop from each pair of highly correlated features are:\n{}'.format(features_to_drop_pairs))
```

    The features to drop from each pair of highly correlated features are:
    ['GarageCars', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF']
    

### Dropping low correlated features with the target variable
- Ill define low correlation as an absolute values of less than 0.05


```python
low_corr_dict = dict()

for feature in numerical_df.columns:
    correlation_with_y = abs(y.corr(numerical_df[feature]))
    
    if correlation_with_y <= 0.05:
        low_corr_dict[feature] = correlation_with_y
        
low_corr_set = set(sorted(low_corr_dict.items(), key=lambda low_corr_dict: low_corr_dict[1]))
low_corr_set
```




    {('3SsnPorch', 0.04458366533574843),
     ('BsmtFinSF2', 0.011378121450215137),
     ('BsmtHalfBath', 0.016844154297359012),
     ('LowQualFinSF', 0.025606130000679548),
     ('MiscVal', 0.021189579640303245),
     ('MoSold', 0.04643224522381936),
     ('YrSold', 0.028922585168730284)}




```python
features_to_drop_target = set()
for feature in low_corr_set:
    features_to_drop_target.add(feature[0])
    
features_to_drop_target
```




    {'3SsnPorch',
     'BsmtFinSF2',
     'BsmtHalfBath',
     'LowQualFinSF',
     'MiscVal',
     'MoSold',
     'YrSold'}



# Feature Selection - Categorical Features

## Chi-Square

lets define :
- Null Hypothesis- The two *Categorical* variables are independent.
- Alternate Hypothesis- The two *Categorical* variables are dependent.


```python
from sklearn.feature_selection import chi2
from scipy import stats
from Helpers import ChiSquare
```


```python
c = ChiSquare(df)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>1456</td>
      <td>60</td>
      <td>RL</td>
      <td>62.0</td>
      <td>7917</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>175000</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>1457</td>
      <td>20</td>
      <td>RL</td>
      <td>85.0</td>
      <td>13175</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>210000</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>1458</td>
      <td>70</td>
      <td>RL</td>
      <td>66.0</td>
      <td>9042</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>Shed</td>
      <td>2500</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>266500</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>1459</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>9717</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>142125</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>1460</td>
      <td>20</td>
      <td>RL</td>
      <td>75.0</td>
      <td>9937</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>147500</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 81 columns</p>
</div>




```python
features_to_drop_chi_withP_val = c.features_to_drop('SalePrice')
features_to_drop_chi_withP_val
```




    [('Electrical', 0.055687991437252254),
     ('Condition2', 0.07598640644469955),
     ('LandContour', 0.08674645041917711),
     ('LandSlope', 0.10508638737793802),
     ('GarageType', 0.13248306385029038),
     ('Alley', 0.20972415885759163),
     ('MiscFeature', 0.23779471590496037),
     ('PoolQC', 0.3007082761743609),
     ('HouseStyle', 0.6482615179447816),
     ('Exterior2nd', 0.8469189757654344),
     ('Fence', 0.9141088546261399),
     ('BsmtFinType1', 0.9728950558884568),
     ('PavedDrive', 0.9912799626956016),
     ('HeatingQC', 0.9995946871632676),
     ('Exterior1st', 0.9999839433628513),
     ('GarageCond', 0.999985676535125),
     ('BldgType', 0.9999860714473023),
     ('BsmtFinType2', 0.9999995053342521),
     ('Utilities', 1.0),
     ('Condition1', 1.0),
     ('RoofStyle', 1.0),
     ('RoofMatl', 1.0),
     ('Functional', 1.0)]



## Summary

- Lets combine all the features that did not meet any one of these tests


```python
features_to_drop_chi = set()
for feature in features_to_drop_chi_withP_val:
    features_to_drop_chi.add(feature[0])
    
```


```python
overall_features_to_drop = features_dropped_var.union(features_dropped_const, features_to_drop_pairs, features_to_drop_target,features_to_drop_chi )
print('The features that did not meet any one of the feature selection tests are: \n{}'.format(overall_features_to_drop))
```

    The features that did not meet any one of the feature selection tests are: 
    {'Fence', 'GarageYrBlt', 'BsmtFinType1', 'GarageCond', 'BsmtFinType2', 'PoolQC', 'BsmtFinSF2', 'Exterior2nd', 'MiscFeature', 'LandContour', 'HeatingQC', 'GarageType', 'RoofStyle', 'KitchenAbvGr', 'HouseStyle', 'Condition2', 'Alley', 'Condition1', 'PoolArea', 'Functional', 'PavedDrive', 'GarageCars', 'LowQualFinSF', 'MoSold', '1stFlrSF', 'RoofMatl', 'Electrical', 'Exterior1st', 'TotRmsAbvGrd', 'LandSlope', 'Utilities', 'BsmtHalfBath', 'BldgType', '3SsnPorch', 'YrSold', 'MiscVal'}
    


```python

```
