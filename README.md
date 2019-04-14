# Understanding Seattle Airbnb Data
The Ananconda notebook corresponding to a data science blog analyzing AirBnB data.

## Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Directory Structure](#directoryStructure)
4. [Results](#results)

## Installation <a name="installation"></a>

This project was written in Python 3.6, using Jupyter Notebook on Anaconda. The relevant Python packages for this project are as follows:

- numpy
- pandas
- matplotlib
- IPython (display module)
- collections
- itertools (chain module)
- sklearn
- time
- sklearn.model_selection (train_test_split module)
- sklearn.preprocessing (Imputer and StandardScaler modules)
- sklearn.metrics (mean_squared_error module)
- sklearn.linear_model (LinearRegression and Logistic regressionmodule)
- sklearn.tree (DecisionTreeRegressor module)
- sklearn.svm (SVM module)
- seaborn

The Matplotlib Basemap Toolkit was used outside of the Jupyter notebook in a python file.

## Project Motivation <a name="motivation"></a>

The notebook tries to answer 3 business questions about Seattle Airbnb rentals:

- What drives higher prices ?
- When are the highest prices paid ?
- Where are the highest prices paid ?

## Directory Structure <a name="directoryStructure"></a>

Root/
├── Understanding Seattle Airbnb Data.ipynb  
├── helper.py  
├── README.md  
└── Data/  
|   ├── calendar.csv  
|   ├──	listings.csv  
|   └── reviews.csv  
└── Images/  
    ├── amenities.png  
    ├──	price_histogram.png  
    └── price_time_series.png  
    └── seattle_prices.png  

## Results <a name="results"></a>
There is a large range of prices Seattle home owners demand for their services. Some amenities seem to have a large impact on prices despite being quite cheap. These amenities are a prudent investment. There exists a significant seasonal influence on prices. Special events like trade shows drive prices upwards for a short amount of time. Most offerings are located near the city center, with a drop of prices and the number of offerings farther away.
