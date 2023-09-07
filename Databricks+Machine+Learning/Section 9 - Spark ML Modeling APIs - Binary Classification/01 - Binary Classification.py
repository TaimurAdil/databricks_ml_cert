# Databricks notebook source
# MAGIC %md
# MAGIC - The Spark MLlib Pipelines API provides higher-level API built on top of DataFrames for constructing ML pipelines.
# MAGIC You can read more about the Pipelines API in the [MLlib programming guide](https://spark.apache.org/docs/latest/ml-guide.html).
# MAGIC
# MAGIC - In this notebook we will build a binary classification application using the Apache Spark MLlib Pipelines API
# MAGIC
# MAGIC - **Binary Classification** is the task of predicting a binary label.
# MAGIC For example, is an email spam or not spam? Should I show this ad to this user or not? Will it rain tomorrow or not?

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1: Dataset Overview

# COMMAND ----------

# MAGIC %md
# MAGIC The Adult dataset is publicly available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult).
# MAGIC This data derives from census data and consists of information about 48842 individuals and their annual income.
# MAGIC You can use this information to predict if an individual earns **<=50K or >50k** a year.
# MAGIC The dataset consists of both numeric and categorical variables.
# MAGIC
# MAGIC Attribute Information:
# MAGIC
# MAGIC - age: continuous
# MAGIC - workclass: Private,Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
# MAGIC - fnlwgt: continuous
# MAGIC - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc...
# MAGIC - education-num: continuous
# MAGIC - marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent...
# MAGIC - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners...
# MAGIC - relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
# MAGIC - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
# MAGIC - sex: Female, Male
# MAGIC - capital-gain: continuous
# MAGIC - capital-loss: continuous
# MAGIC - hours-per-week: continuous
# MAGIC - native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany...
# MAGIC
# MAGIC Target/Label: - <=50K, >50K
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: Load Data

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/adult/adult.data

# COMMAND ----------

from pyspark.sql.types import DoubleType, StringType, StructField, StructType

schema = StructType([
  StructField("age", DoubleType(), False),
  StructField("workclass", StringType(), False),
  StructField("fnlwgt", DoubleType(), False),
  StructField("education", StringType(), False),
  StructField("education_num", DoubleType(), False),
  StructField("marital_status", StringType(), False),
  StructField("occupation", StringType(), False),
  StructField("relationship", StringType(), False),
  StructField("race", StringType(), False),
  StructField("sex", StringType(), False),
  StructField("capital_gain", DoubleType(), False),
  StructField("capital_loss", DoubleType(), False),
  StructField("hours_per_week", DoubleType(), False),
  StructField("native_country", StringType(), False),
  StructField("income", StringType(), False)
])

dataset = spark.read.format("csv").schema(schema).load("/databricks-datasets/adult/adult.data")

# COMMAND ----------

display(dataset)

# COMMAND ----------

type(dataset)

# COMMAND ----------

cols = dataset.columns
print(cols)

# COMMAND ----------

print("Total number of observations in the dataset are: ", dataset.count())
print("Total number of columns in the dataset are: ", len(cols))

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3: Data Preprocessing & Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC **Steps:**
# MAGIC - Encoding the categorical columns (Features & Label Column)
# MAGIC - Transform features into a vector
# MAGIC - Run stages as a Pipeline
# MAGIC - Keep relevant columns
# MAGIC - Split data into training and test sets

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC To use algorithms like Logistic Regression, you must first convert the categorical variables in the dataset into numeric variables.
# MAGIC There are two ways to do this.
# MAGIC
# MAGIC * Category Indexing
# MAGIC
# MAGIC   This is basically assigning a numeric value to each category from {0, 1, 2, ...numCategories-1}.
# MAGIC   This introduces an implicit ordering among your categories, and is more suitable for ordinal variables (eg: Poor: 0, Average: 1, Good: 2)
# MAGIC
# MAGIC * One-Hot Encoding
# MAGIC
# MAGIC   This converts categories into binary vectors with at most one nonzero value (eg: (Blue: [1, 0]), (Green: [0, 1]), (Red: [0, 0]))
# MAGIC
# MAGIC In this notebook uses a combination of [StringIndexer] and, depending on your Spark version, either [OneHotEncoder] or [OneHotEncoderEstimator] to convert the categorical variables.
# MAGIC `OneHotEncoder` and `OneHotEncoderEstimator` return a [SparseVector]. 
# MAGIC
# MAGIC Since there is more than one stage of feature transformations, use a [Pipeline] to tie the stages together.
# MAGIC This simplifies the code.
# MAGIC
# MAGIC [StringIndexer]: http://spark.apache.org/docs/latest/ml-features.html#stringindexer
# MAGIC [OneHotEncoderEstimator]: https://spark.apache.org/docs/2.4.5/api/python/pyspark.ml.html?highlight=one%20hot%20encoder#pyspark.ml.feature.OneHotEncoderEstimator
# MAGIC [SparseVector]: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.linalg.SparseVector.html#pyspark.ml.linalg.SparseVector
# MAGIC [Pipeline]: https://spark.apache.org/docs/latest/ml-pipeline.html#ml-pipelines
# MAGIC [OneHotEncoder]: https://spark.apache.org/docs/latest/ml-features.html#onehotencoder

# COMMAND ----------

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from distutils.version import LooseVersion

# COMMAND ----------

categoricalColumns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
print(categoricalColumns)

# COMMAND ----------

stages = [] # stages in Pipeline

for categoricalCol in categoricalColumns:
    
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
        from pyspark.ml.feature import OneHotEncoderEstimator
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    else:
        from pyspark.ml.feature import OneHotEncoder
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    
    # Add stages, these are not run here, but will run all at once later on
    stages += [stringIndexer, encoder]

# COMMAND ----------

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="income", outputCol="label")
stages += [label_stringIdx]

# COMMAND ----------

# Transform all features into a vector using VectorAssembler
numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

# Run the stages as a Pipeline
partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(dataset)
preppedDataDF = pipelineModel.transform(dataset)

# COMMAND ----------

preppedDataDF.count()

# COMMAND ----------

preppedDataDF.columns

# COMMAND ----------

len(preppedDataDF.columns)

# COMMAND ----------

# Keep relevant columns
selectedcols = ["label", "features"] + cols
dataset = preppedDataDF.select(selectedcols)
display(dataset)

# COMMAND ----------

len(dataset.columns)

# COMMAND ----------

# Randomly split data into training and test sets, set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 4: Train and Evaluate Models

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we will try some of the Binary Classification algorithms available in the Pipelines API.
# MAGIC
# MAGIC **Steps to build the models:**
# MAGIC - Create initial model using the training set
# MAGIC - Tune parameters with a `ParamGrid` and 5-fold Cross Validation
# MAGIC - Evaluate the best model obtained from the Cross Validation using the test set
# MAGIC
# MAGIC We will use the `BinaryClassificationEvaluator` to evaluate the models, which uses [areaUnderROC] as the default metric.
# MAGIC
# MAGIC [areaUnderROC]: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Logistic Regression without hyperparameter tuning
# MAGIC
# MAGIC You can read more about [Logistic Regression] from the [classification and regression] section of MLlib Programming Guide.
# MAGIC In the Pipelines API, you can now perform Elastic-Net Regularization with Logistic Regression, as well as other linear methods.
# MAGIC
# MAGIC [classification and regression]: https://spark.apache.org/docs/latest/ml-classification-regression.html
# MAGIC [Logistic Regression]: https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Create initial logistic regression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# Train the model with training data
lrModel =  lr.fit(trainingData)

# COMMAND ----------

# Make the predictions with test data
predictions = lrModel.transform(testData)

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
# You can select any columns in the above schema to view as well
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# Evaluate model
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

# Get the evaluation metric name
evaluator.getMetricName()

# COMMAND ----------

print(lr.explainParams())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression with hyperparameter tuning

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())

# COMMAND ----------

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(trainingData)
# this will likely take a fair amount of time because of the amount of models that we're creating and testing

# COMMAND ----------

# Make the predictions with test data
predictions = cvModel.transform(testData)

# COMMAND ----------

evaluator.evaluate(predictions)

# COMMAND ----------

# Get the evaluation metric name
evaluator.getMetricName()

# COMMAND ----------

# Model Intercepts
print("Model Intercept : ", cvModel.bestModel.intercept)

# COMMAND ----------

# Model Weights
weights = cvModel.bestModel.coefficients
weights = [(float(w),) for w in weights]  # convert numpy type to float, and to tuple
weightsDF = spark.createDataFrame(weights, ["Feature Weight"])
display(weightsDF)

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
# You can select any columns in the above schema to view as well
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Trees
# MAGIC
# MAGIC You can read more about [Decision Trees](http://spark.apache.org/docs/latest/mllib-decision-tree.html) in the Spark MLLib Programming Guide.
# MAGIC The Decision Trees algorithm is popular because it handles categorical
# MAGIC data and works out of the box with multiclass classification tasks.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision Trees without hyperparameter tuning

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

# Create initial Decision Tree Model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=3)

# Train model with Training Data
dtModel = dt.fit(trainingData)

# COMMAND ----------

# Extract the number of nodes and tree depth of decision tree model 
print("numNodes = ", dtModel.numNodes)
print("depth = ", dtModel.depth)

# COMMAND ----------

display(dtModel)

# COMMAND ----------

# Make predictions on test data using the Transformer.transform() method.
predictions = dtModel.transform(testData)

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# Evaluate the Decision Tree model
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision Trees with hyperparameter tuning

# COMMAND ----------

# Create ParamGrid for Cross Validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [1, 2, 6, 10])
             .addGrid(dt.maxBins, [20, 40, 80])
             .build())

# COMMAND ----------

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(trainingData)

# COMMAND ----------

# Extract the number of nodes and tree depth of decision tree model 

print("numNodes = ", cvModel.bestModel.numNodes)
print("depth = ", cvModel.bestModel.depth)

# COMMAND ----------

# Use test set to measure the accuracy of the model on new data
predictions = cvModel.transform(testData)

# COMMAND ----------

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
evaluator.evaluate(predictions)

# COMMAND ----------

# View Best model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest
# MAGIC
# MAGIC Random Forests uses an ensemble of trees to improve model accuracy.
# MAGIC You can read more about [Random Forest] from the [classification and regression] section of MLlib Programming Guide.
# MAGIC
# MAGIC [classification and regression]: https://spark.apache.org/docs/latest/ml-classification-regression.html
# MAGIC [Random Forest]: https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forests

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random forest without hyperparameter tuning

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# Create an initial RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Train model with Training Data
rfModel = rf.fit(trainingData)

# COMMAND ----------

# Make predictions on test data using the Transformer.transform() method.
predictions = rfModel.transform(testData)

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# Evaluate model
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random forest with hyperparameter tuning

# COMMAND ----------

# Create ParamGrid for Cross Validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 20])
             .build())

# COMMAND ----------

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(trainingData)

# COMMAND ----------

# Use the test set to measure the accuracy of the model on new data
predictions = cvModel.transform(testData)

# COMMAND ----------

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
evaluator.evaluate(predictions)

# COMMAND ----------

# View Best model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 5: Make Predictions

# COMMAND ----------

bestModel = cvModel.bestModel

# COMMAND ----------

print(bestModel)

# COMMAND ----------

# Generate predictions for entire dataset
finalPredictions = bestModel.transform(dataset)

# COMMAND ----------

# Evaluate best model
evaluator.evaluate(finalPredictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Predictions grouped by age and occupation

# COMMAND ----------

finalPredictions.createOrReplaceTempView("finalPredictions")

# COMMAND ----------

# Predictions grouped by age

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT age, prediction, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY age, prediction
# MAGIC ORDER BY age

# COMMAND ----------

# Predictions grouped by occupation

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT occupation, prediction, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY occupation, prediction
# MAGIC ORDER BY occupation
