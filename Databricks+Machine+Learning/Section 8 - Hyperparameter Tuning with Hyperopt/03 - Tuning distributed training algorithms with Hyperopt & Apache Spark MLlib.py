# Databricks notebook source
# MAGIC %md 
# MAGIC Databricks Runtime for Machine Learning includes,
# MAGIC 1) [Hyperopt](https://github.com/hyperopt/hyperopt): A library for ML hyperparameter tuning in Python
# MAGIC 2) [Apache Spark MLlib](https://spark.apache.org/docs/latest/ml-guide.html): A library of distributed algorithms for training ML models (also called as "Spark ML")
# MAGIC
# MAGIC **In this notebook we will learn to use them together:** We have distributed ML workloads in Python for which we want to tune hyperparameters
# MAGIC
# MAGIC This notebook includes two sections:
# MAGIC * **Part 1: Run distributed training using MLlib:** In this section we will do the MLlib model training without hyperparameter tuning
# MAGIC * **Part 2: Use Hyperopt to tune hyperparameters in the distributed training workflow:** Here we will wrap the MLlib code with Hyperopt for tuning

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1: Run distributed training using MLlib

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data
# MAGIC - **MNIST handwritten digit recognition dataset:** A classic dataset of handwritten digits that is commonly used for training and benchmarking ML algorithms
# MAGIC - It consists of 60,000 training images and 10,000 test images, each of which is a 28x28 pixel grayscale image of a handwritten digit
# MAGIC - The digits in the dataset are labeled from 0 to 9, and the task is to classify a given image as one of these 10 classes
# MAGIC  - It is stored in the popular LibSVM dataset format, we will load MNIST dataset using MLlib's LibSVM dataset reader utility

# COMMAND ----------

full_training_data = spark.read.format("libsvm").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt")
test_data = spark.read.format("libsvm").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-test.txt")

# Cache data for multiple uses
full_training_data.cache()
test_data.cache()

print(f"There are {full_training_data.count()} training images and {test_data.count()} test images.")

# COMMAND ----------

# Randomly split full_training data for tuning
training_data, validation_data = full_training_data.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

display(training_data)

# COMMAND ----------

display(validation_data)

# COMMAND ----------

display(test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a function to train a model
# MAGIC
# MAGIC We will define a function to train a decision tree. Wrapping the training code in a function is important for passing the function to Hyperopt for tuning later.
# MAGIC
# MAGIC **Details:** The tree algorithm needs to know that the labels are categories 0-9, rather than continuous values. This example uses the `StringIndexer` class to do this.  A `Pipeline` ties this feature preprocessing together with the tree algorithm.  ML Pipelines are tools Spark provides for piecing together Machine Learning algorithms into workflows.

# COMMAND ----------

import mlflow

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer

# COMMAND ----------

# MLflow autologging for `pyspark.ml` requires MLflow version 1.17.0 or above.
# This try-except logic allows the notebook to run with older versions of MLflow.
try:
  import mlflow.pyspark.ml
  mlflow.pyspark.ml.autolog()
except:
  print(f"Your version of MLflow ({mlflow.__version__}) does not support pyspark.ml for autologging. To use autologging, upgrade your MLflow client version or use Databricks Runtime for ML 8.3 or above.")

# COMMAND ----------

def train_tree(minInstancesPerNode, maxBins):
  '''
  This train() function:
   - takes hyperparameters as inputs (for tuning later)
   - returns the F1 score on the validation dataset

  Wrapping code as a function makes it easier to reuse the code later with Hyperopt.
  '''
  # Use MLflow to track training.
  # Specify "nested=True" since this single model will be logged as a child run of Hyperopt's run.
  with mlflow.start_run(nested=True):
    
    # StringIndexer: Read input column "label" (digits) and annotate them as categorical values.
    indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
    
    # DecisionTreeClassifier: Learn to predict column "indexedLabel" using the "features" column.
    dtc = DecisionTreeClassifier(labelCol="indexedLabel",
                                 minInstancesPerNode=minInstancesPerNode,
                                 maxBins=maxBins)
    
    # Chain indexer and dtc together into a single ML Pipeline.
    pipeline = Pipeline(stages=[indexer, dtc])
    model = pipeline.fit(training_data)

    # Define an evaluation metric and evaluate the model on the validation dataset.
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="f1")
    predictions = model.transform(validation_data)
    validation_metric = evaluator.evaluate(predictions)
    mlflow.log_metric("val_f1_score", validation_metric)

  return model, validation_metric

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a decision tree classifier

# COMMAND ----------

initial_model, val_metric = train_tree(minInstancesPerNode=200, maxBins=2)
print(f"The trained decision tree achieved an F1 score of {val_metric} on the validation data")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: Use Hyperopt to tune hyperparameters
# MAGIC
# MAGIC In this section, you create the Hyperopt workflow. 
# MAGIC * Define a function to minimize
# MAGIC * Define a search space over hyperparameters
# MAGIC * Specify the search algorithm and use `fmin()` to tune the model
# MAGIC
# MAGIC For more information about the Hyperopt APIs, see the [Hyperopt documentation](http://hyperopt.github.io/hyperopt/).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define a function to minimize
# MAGIC
# MAGIC * Input: hyperparameters
# MAGIC * Internally: Reuse the training function defined above.
# MAGIC * Output: loss

# COMMAND ----------

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

def train_with_hyperopt(params):
  """
  An example train method that calls into MLlib.
  This method is passed to hyperopt.fmin().
  
  :param params: hyperparameters as a dict. Its structure is consistent with how search space is defined. See below.
  :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
  """
  # For integer parameters, make sure to convert them to int type if Hyperopt is searching over a continuous range of values.
  minInstancesPerNode = int(params['minInstancesPerNode'])
  maxBins = int(params['maxBins'])

  model, f1_score = train_tree(minInstancesPerNode, maxBins)
  
  # Hyperopt expects you to return a loss (for which lower is better), so take the negative of the f1_score (for which higher is better).
  loss = - f1_score
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the search space over hyperparameters
# MAGIC
# MAGIC This example tunes two hyperparameters: `minInstancesPerNode` and `maxBins`. See the [Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions) for details on defining a search space and parameter expressions.

# COMMAND ----------

import numpy as np
space = {
  'minInstancesPerNode': hp.uniform('minInstancesPerNode', 10, 200),
  'maxBins': hp.uniform('maxBins', 2, 32),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select the search algorithm

# COMMAND ----------

# MAGIC %md
# MAGIC - You must also specify which search algorithm to use. The two main choices are:
# MAGIC   - `hyperopt.tpe.suggest`: Tree of Parzen Estimators, a Bayesian approach which iteratively and adaptively selects new hyperparameter settings to explore based on previous results
# MAGIC   - `hyperopt.rand.suggest`: Random search, a non-adaptive approach that randomly samples the search space

# COMMAND ----------

algo = tpe.suggest

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the tuning algorithm with Hyperopt fmin()

# COMMAND ----------

# MAGIC %md 
# MAGIC **Important:**  
# MAGIC When using Hyperopt with MLlib and other distributed training algorithms, do not pass a `trials` argument to `fmin()`. When you do not include the `trials` argument, Hyperopt uses the default `Trials` class, which runs on the cluster driver. Hyperopt needs to evaluate each trial on the driver node so that each trial can initiate distributed training jobs.  
# MAGIC
# MAGIC Do not use the `SparkTrials` class with MLlib. `SparkTrials` is designed to distribute trials for algorithms that are not themselves distributed. MLlib uses distributed computing already and is not compatible with `SparkTrials`.

# COMMAND ----------

with mlflow.start_run():
  best_params = fmin(
    fn=train_with_hyperopt,
    space=space,
    algo=algo,
    max_evals=8
  )

# COMMAND ----------

# Best hyperparametrs
print(best_params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrain the model on training dataset

# COMMAND ----------

best_minInstancesPerNode = int(best_params['minInstancesPerNode'])
best_maxBins = int(best_params['maxBins'])

final_model, val_f1_score = train_tree(best_minInstancesPerNode, best_maxBins)

# COMMAND ----------

print(f"The retrained decision tree achieved an F1 score of {val_f1_score} on the validation data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use test dataset to compare evaluation metrics for the initial and "best" model

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="f1")

initial_model_test_metric = evaluator.evaluate(initial_model.transform(test_data))
final_model_test_metric = evaluator.evaluate(final_model.transform(test_data))

print(f"On the test data, the initial (untuned) model achieved F1 score {initial_model_test_metric}, and the final (tuned) model achieved {final_model_test_metric}.")
