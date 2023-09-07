# Databricks notebook source
# MAGIC %md
# MAGIC **Automated MLflow tracking in MLlib:**
# MAGIC
# MAGIC - MLflow provides automated tracking for model tuning with MLlib
# MAGIC - With automated MLflow tracking, when you run tuning code using `CrossValidator` or `TrainValidationSplit`, the specified hyperparameters and evaluation metrics are automatically logged in MLflow
# MAGIC - It makes easy to identify the optimal model, without automated MLflow tracking you must make explicit API calls to log to MLflow
# MAGIC
# MAGIC **In this notebook we will learn:** Automated MLflow tracking with MLlib. 
# MAGIC
# MAGIC - In this notebook we will use the PySpark classes `DecisionTreeClassifier` and `CrossValidator` to train and tune a model. MLflow automatically tracks the learning process and saves the results of each run, So you can examine the hyperparameters to understand the impact of each one on the model's performance and find the optimal settings
# MAGIC
# MAGIC **Dataset:** MNIST handwritten digit recognition dataset, which is included with Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1: Train model without cross validation 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the training and test datasets
# MAGIC - **MNIST handwritten digit recognition dataset:** A classic dataset of handwritten digits that is commonly used for training and benchmarking ML algorithms
# MAGIC - It consists of 60,000 training images and 10,000 test images, each of which is a 28x28 pixel grayscale image of a handwritten digit
# MAGIC - The digits in the dataset are labeled from 0 to 9, and the task is to classify a given image as one of these 10 classes
# MAGIC  - It is stored in the popular LibSVM dataset format, we will load MNIST dataset using MLlib's LibSVM dataset reader utility

# COMMAND ----------

training = spark.read.format("libsvm").option("numFeatures", "784").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt")
test = spark.read.format("libsvm").option("numFeatures", "784").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-test.txt")

# Cache data for multiple uses
training.cache()
test.cache()

print("There are {} training images and {} test images.".format(training.count(), test.count()))

# COMMAND ----------

display(training)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the required classes

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the ML pipeline

# COMMAND ----------

# StringIndexer: Convert the input column "label" (digits) to categorical values
indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
# DecisionTreeClassifier: Learn to predict column "indexedLabel" using the "features" column
dtc = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features", maxBins=8, maxDepth=4)
# Chain indexer + dtc together into a single ML Pipeline
pipeline = Pipeline(stages=[indexer, dtc])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model and make predictions

# COMMAND ----------

# Fit the pipeline on the training dataset
model = pipeline.fit(training)

# Evaluate the model on the test dataset
predictions = model.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an evaluator

# COMMAND ----------

# Create an evaluator using "weightedPrecision".
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="weightedPrecision")
test_metric = evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the metric and parameters

# COMMAND ----------

# Log the model and evaluation metric in MLflow
with mlflow.start_run():
  mlflow.spark.log_model(spark_model=model, artifact_path='best-model')
  mlflow.log_metric('test_' + evaluator.getMetricName(), test_metric)
  
  # Log all the parameters in MLflow
  params = dtc.extractParamMap()
  for param_name, param_value in params.items():
    mlflow.log_param(param_name.name, param_value)
    print(f"{param_name.name}: {param_value}")

# Print the evaluation metric
print("Test Weighted Precision:", test_metric)

# COMMAND ----------

# Retrieve the maxDepth and maxBins parameters from the fitted DecisionTreeClassifier
max_depth = model.stages[-1].getMaxDepth()
max_bins = model.stages[-1].getMaxBins()

# Print the values of maxDepth and maxBins
print("maxDepth:", max_depth)
print("maxBins:", max_bins)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: Train model with cross validation 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the training and test datasets

# COMMAND ----------

training = spark.read.format("libsvm").option("numFeatures", "784").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt")
test = spark.read.format("libsvm").option("numFeatures", "784").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-test.txt")

# Cache data for multiple uses
training.cache()
test.cache()

print("There are {} training images and {} test images.".format(training.count(), test.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the required classes

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# COMMAND ----------

# MAGIC %md ## Define the ML pipeline 
# MAGIC
# MAGIC In this example, we have to do some preprocessing of the data before we can use the data to train a model. MLlib provides **pipelines** that allows us to combine multiple steps into a single workflow
# MAGIC
# MAGIC In this example, we will build a two-step pipeline:
# MAGIC 1. `StringIndexer` converts the labels from numeric values to categorical values. 
# MAGIC 2. `DecisionTreeClassifier` trains a decision tree that can predict the "label" column based on the data in the "features" column.
# MAGIC
# MAGIC For more information:  
# MAGIC [Pipelines](http://spark.apache.org/docs/latest/ml-pipeline.html#ml-pipelines)

# COMMAND ----------

# StringIndexer: Convert the input column "label" (digits) to categorical values
indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
# DecisionTreeClassifier: Learn to predict column "indexedLabel" using the "features" column
dtc = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features", maxBins=8, maxDepth=4)
# Chain indexer + dtc together into a single ML Pipeline
pipeline = Pipeline(stages=[indexer, dtc])

# COMMAND ----------

# MAGIC %md ## Run the cross-validation 
# MAGIC
# MAGIC We have defined the pipeline, now we can run the cross validation to tune the model's hyperparameters. During this process, MLflow automatically tracks the models produced by `CrossValidator`, along with their evaluation metrics. This allows you to investigate how specific hyperparameters affect the model's performance.
# MAGIC
# MAGIC In this example, we will examine two hyperparameters in the cross-validation:
# MAGIC
# MAGIC * `maxDepth`. This parameter determines how deep, and thus how large, the tree can grow. 
# MAGIC * `maxBins`. For efficient distributed training of Decision Trees, MLlib discretizes (or "bins") continuous features into a finite number of values. The number of bins is controlled by `maxBins`. In this example, the number of bins corresponds to the number of grayscale levels; `maxBins=2` turns the images into black and white images.
# MAGIC
# MAGIC For more information:  
# MAGIC [maxBins](https://spark.apache.org/docs/latest/mllib-decision-tree.html#split-candidates)  
# MAGIC [maxDepth](https://spark.apache.org/docs/latest/mllib-decision-tree.html#stopping-rule)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create an evaluator

# COMMAND ----------

# Create an evaluator using "weightedPrecision".
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="weightedPrecision")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import CrossValidator, ParamGridBuilder Classes

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the parameter grid

# COMMAND ----------

# Define the parameter grid to examine
grid = ParamGridBuilder() \
  .addGrid(dtc.maxDepth, [2, 3, 4, 5, 6, 7, 8]) \
  .addGrid(dtc.maxBins, [2, 4, 8]) \
  .build()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a cross validator

# COMMAND ----------

cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=grid, numFolds=3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run CrossValidator

# COMMAND ----------

# MAGIC %md Run `CrossValidator`.  If an MLflow tracking server is available, `CrossValidator` automatically logs each run to MLflow, along with the evaluation metric calculated on the held-out data, under the current active run. If no run is active, a new one is created. 

# COMMAND ----------

# Explicitly creating a new run, it will allows this cell to be run multiple times
# If you omit mlflow.start_run(), then this cell could run once, but a second run would hit conflicts when attempting to overwrite the first run

import mlflow
import mlflow.spark

with mlflow.start_run():
  # Run the cross validation on the training dataset. The cv.fit() call returns the best model it found.
  cvModel = cv.fit(training)

  # Retrieve the best model's parameters
  bestParams = cvModel.bestModel.stages[-1].extractParamMap()

  # Evaluate the best model's performance on the test dataset and log the result.
  test_metric = evaluator.evaluate(cvModel.transform(test))
  mlflow.log_metric('test_' + evaluator.getMetricName(), test_metric)

  # Log the best model.
  mlflow.spark.log_model(spark_model=cvModel.bestModel, artifact_path='best-model')

  # Log all the parameters in MLflow
  params = dtc.extractParamMap()
  for param_name, param_value in params.items():
    mlflow.log_param(param_name.name, param_value)
    print(f"{param_name.name}: {param_value}")

  # Print the evaluation metric
  print("Test Weighted Precision:", test_metric)

# COMMAND ----------

# MAGIC %md
# MAGIC - maxDepth: 4
# MAGIC - maxBins: 8

# COMMAND ----------

  # Print the best parameters
  print("Best parameters: ")
  print("maxDepth =", bestParams[dtc.maxDepth])
  print("maxBins =", bestParams[dtc.maxBins])
