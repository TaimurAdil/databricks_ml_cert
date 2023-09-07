# Databricks notebook source
# MAGIC %md
# MAGIC # Import required packages and load dataset

# COMMAND ----------

import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

import mlflow

# COMMAND ----------

X, y = fetch_california_housing(return_X_y=True)

# COMMAND ----------

# MAGIC %md
# MAGIC - The California housing dataset is a widely used dataset in machine learning and is available in the scikit-learn library
# MAGIC - It contains information about housing prices in various districts of California. The dataset is often used for regression tasks to predict the median house value in a given district based on several features.
# MAGIC
# MAGIC - **The California housing dataset provides the following information for each district:**
# MAGIC
# MAGIC 1) MedInc: Median income of households in the district.
# MAGIC 2) HouseAge: Median age of houses in the district.
# MAGIC 3) AveRooms: Average number of rooms per house.
# MAGIC 4) AveBedrms: Average number of bedrooms per house.
# MAGIC 5) Population: Total population in the district.
# MAGIC 6) AveOccup: Average number of occupants per house.
# MAGIC 7) Latitude: Latitude of the district's location.
# MAGIC 8) Longitude: Longitude of the district's location.
# MAGIC 9) MedHouseVal: Median value of houses in the district (the target variable).
# MAGIC
# MAGIC - The goal of using this dataset is typically to build a regression model that can predict the median house value based on the given features.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scale the features
# MAGIC

# COMMAND ----------

X.mean(axis=0)

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# COMMAND ----------

scalar = StandardScaler()

# COMMAND ----------

X = scalar.fit_transform(X)

# COMMAND ----------

X.mean(axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert the numeric target column to discrete values

# COMMAND ----------

y_discrete = np.where(y < np.median(y), 0, 1)

# COMMAND ----------

print(y_discrete)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Hyperopt workflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the function to minimize

# COMMAND ----------

def objective(params):
    classifier_type = params['type']
    del params['type']
    if classifier_type == 'svm':
        clf = SVC(**params)
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(**params)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(**params)
    else:
        return 0
    accuracy = cross_val_score(clf, X, y_discrete).mean()
    
    # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the search space over hyperparameters

# COMMAND ----------

search_space = hp.choice('classifier_type', [
    {
        'type': 'svm',
        'C': hp.lognormal('SVM_C', 0, 1.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf'])
    },
    {
        'type': 'rf',
        'max_depth': hp.quniform('max_depth', 2, 5, 1),
        'criterion': hp.choice('criterion', ['gini', 'entropy'])
    },
    {
        'type': 'logreg',
        'C': hp.lognormal('LR_C', 0, 1.0),
        'solver': hp.choice('solver', ['liblinear', 'lbfgs'])
    },
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select the search algorithm
# MAGIC
# MAGIC The two main choices are:
# MAGIC * `hyperopt.tpe.suggest`: Tree of Parzen Estimators, a Bayesian approach that iteratively and adaptively selects new hyperparameter settings to explore based on previous results
# MAGIC * `hyperopt.rand.suggest`: Random search, a non-adaptive approach that samples over the search space

# COMMAND ----------

algo = tpe.suggest

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the tuning algorithm with Hyperopt fmin()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC To distribute tuning, add one more argument to `fmin()`: Class `Trials` & Method `SparkTrials`
# MAGIC
# MAGIC `SparkTrials` takes 2 optional arguments:  
# MAGIC * `parallelism`: Number of models to fit and evaluate concurrently. The default is the number of available Spark task slots.
# MAGIC * `timeout`: Maximum time (in seconds) that `fmin()` can run. The default is no maximum time limit.

# COMMAND ----------

from hyperopt import SparkTrials

# COMMAND ----------

spark_trials = SparkTrials()

# COMMAND ----------

with mlflow.start_run():
  best_results = fmin(
    fn=objective,
    space=search_space,
    algo=algo,
    max_evals=32,
    trials=spark_trials
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Print the hyperparameters that produced the best result

# COMMAND ----------

import hyperopt

# COMMAND ----------

print(hyperopt.space_eval(search_space, best_results))
