# CA04
Creating a random forest model with different classifiers

This program uses a census dataset and builds a random forest model from it to attempt to predict income. After building the initial random forest model we then tried three different classifiers AdaBoost, Gradient Boost, and Extreme Gradient Boost. From these we determines the appropriate amount of estimators and tested out common hyper-parameters on all four of our models to see which one was the best. The best ended up being our extreme gradient boost classifier which had the highest accuracy and AUC. 

The packages we needed are stated below:

import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from google.colab import drive
import sklearn.ensemble
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
!pip install xgboost
import xgboost as xgb

In order to download the data I used google drive and imported drive. We then have to mount our document to our drive. Lastly, in order to read in the data we simply connect it to our Colab Notebooks and then we are ready to connect to our data and perform our analysis.

In order to run the program, simply connect to the necessary data, import the above packages, and run the programs. The programs running the random forest model may take longer than usual to run. The final result will be a database showcasing the accuracy and AUC of each model and classifiers as well as the common hyper parameters that were used (n_estimator: 450, random_state: 101). 
