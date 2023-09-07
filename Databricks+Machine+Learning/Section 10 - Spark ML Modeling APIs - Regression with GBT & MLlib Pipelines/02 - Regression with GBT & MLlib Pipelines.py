# Databricks notebook source
# MAGIC %md
# MAGIC # Part 1: Business Problem

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC - In this notebook we will use a bike-sharing dataset to **illustrate MLlib pipelines and the gradient-boosted trees ML algorithm.**
# MAGIC
# MAGIC - Documentation: [Gradient-Boosted Trees (GBT)](https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier) algorithm
# MAGIC
# MAGIC - The challenge is to predict the number of bicycle rentals per hour based on the features available in the dataset such as day of the week, weather, season, and so on...
# MAGIC
# MAGIC - Demand prediction is a common problem across businesses; good predictions allow a business or service to optimize inventory and to match supply and demand to make customers happy and maximize profitability. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: Data Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the dataset

# COMMAND ----------

# MAGIC %md
# MAGIC - The dataset is from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) and is provided with Databricks Runtime. The dataset includes information about bicycle rentals from the Capital bikeshare system in 2011 and 2012

# COMMAND ----------

# Load the data
df = spark.read.csv("/databricks-datasets/bikeSharing/data-001/hour.csv", header="true", inferSchema="true")

# Cache the DataFrame in memory
df.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data description

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The following columns are included in the dataset:
# MAGIC
# MAGIC **Index column**:
# MAGIC * instant: record index
# MAGIC
# MAGIC **Feature columns**:
# MAGIC * dteday: date
# MAGIC * season: season (1:spring, 2:summer, 3:fall, 4:winter)
# MAGIC * yr: year (0:2011, 1:2012)
# MAGIC * mnth: month (1 to 12)
# MAGIC * hr: hour (0 to 23)
# MAGIC * holiday: 1 if holiday, 0 otherwise
# MAGIC * weekday: day of the week (0 to 6)
# MAGIC * workingday: 0 if weekend or holiday, 1 otherwise
# MAGIC * weathersit: (1:clear, 2:mist or clouds, 3:light rain or snow, 4:heavy rain or snow)  
# MAGIC * temp: normalized temperature in Celsius  
# MAGIC * atemp: normalized feeling temperature in Celsius  
# MAGIC * hum: normalized humidity  
# MAGIC * windspeed: normalized wind speed  
# MAGIC
# MAGIC **Label columns**:
# MAGIC * casual: count of casual users
# MAGIC * registered: count of registered users
# MAGIC * cnt: count of total rental bikes including both casual and registered

# COMMAND ----------

display(df)

# COMMAND ----------

print("This dataset has %d rows." % df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drop columns

# COMMAND ----------

# MAGIC %md
# MAGIC - This dataset is well prepared for ML algorithms
# MAGIC - The numeric input columns (temp, atemp, hum, and windspeed) are normalized, categorial values (season, yr, mnth, hr, holiday, weekday, workingday, weathersit) are converted to indices, and all of the columns except for the date (`dteday`) are numeric.
# MAGIC
# MAGIC - The goal is to predict the count of bike rentals (the `cnt` column). Reviewing the dataset, you can see that some columns contain duplicate information. For example, the `cnt` column equals the sum of the `casual` and `registered` columns. You should remove the `casual` and `registered` columns from the dataset. The index column `instant` is also not useful as a predictor.
# MAGIC - You can also delete the column `dteday`, as this information is already included in the other date-related columns `yr`, `mnth`, and `weekday`. 

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.drop("instant").drop("dteday").drop("casual").drop("registered")
display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split data into training and test sets

# COMMAND ----------

# Split the dataset randomly into 70% for training and 30% for testing
train, test = df.randomSplit([0.7, 0.3], seed = 0)
print("There are %d training examples and %d test examples." % (train.count(), test.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize the data

# COMMAND ----------

display(train.select("hr", "cnt"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3: Train the machine learning pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Most MLlib algorithms require a single input column containing a vector of features and a single target column.  
# MAGIC
# MAGIC **In this example, we will create a pipeline using the following functions:**
# MAGIC * `VectorAssembler`: Assembles the feature columns into a feature vector  
# MAGIC * `VectorIndexer`: Identifies columns that should be treated as categorical
# MAGIC * `GBTRegressor`: Uses the [Gradient-Boosted Trees (GBT)](https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier) algorithm to learn how to predict rental counts from the feature vectors
# MAGIC * `CrossValidator`: Evaluate and tune ML model. [hyperparameter tuning in Spark](https://spark.apache.org/docs/latest/ml-tuning.html)
# MAGIC
# MAGIC For more information:  
# MAGIC [VectorAssembler](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler)  
# MAGIC [VectorIndexer](https://spark.apache.org/docs/latest/ml-features.html#vectorindexer)  

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create VectorAssembler and VectorIndexer steps

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer

# Remove the target column from the input feature set
featuresCols = df.columns
featuresCols.remove('cnt')

# vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures"
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")

# vectorIndexer identifies categorical features and indexes them, and creates a new column "features" 
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the model

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor

# Define the model training stage of the pipeline 
gbt = GBTRegressor(labelCol="cnt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap the model in CrossValidator stage

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# Define a grid of hyperparameters to test:
#  - maxDepth: maximum depth of each decision tree 
#  - maxIter: iterations, or the total number of trees 
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 100])\
  .build()

# Define an evaluation metric
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())

# Declare the CrossValidator, which performs the model tuning
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create pipeline

# COMMAND ----------

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the pipeline

# COMMAND ----------

pipelineModel = pipeline.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 4: Make predictions and evaluate results

# COMMAND ----------

predictions = pipelineModel.transform(test)

# COMMAND ----------

display(predictions.select("cnt", "prediction", *featuresCols))

# COMMAND ----------

rmse = evaluator.evaluate(predictions)
print("RMSE on the test set: %g" %rmse)

# COMMAND ----------

import pyspark.sql.functions as F
predictions_with_residuals = predictions.withColumn("residual", (F.col("cnt") - F.col("prediction")))
display(predictions_with_residuals.agg({'residual': 'mean'}))

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 5: Improving the model

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Suggestions for improving this model:**
# MAGIC
# MAGIC 1) The count of rentals is the sum of `registered` and `casual` rentals. These two counts may have different behavior, as frequent cyclists and casual cyclists may rent bikes for different reasons. Try training one GBT model for `registered` and one for `casual`, and then add their predictions together to get the full prediction.
# MAGIC
# MAGIC 2) For efficiency, we used only a few hyperparameter settings. You might be able to improve the model by testing more settings. A good start is to increase the number of trees by setting `maxIter=200`; this takes longer to train but might more accurate.
# MAGIC
# MAGIC 3) We used the dataset features as-is, but you might be able to improve performance with some feature engineering.  
# MAGIC  For example, the weather might have more of an impact on the number of rentals on weekends and holidays than on workdays. You could try creating a new feature by combining those two columns.  MLlib provides a suite of feature transformers; find out more in the [ML guide](http://spark.apache.org/docs/latest/ml-features.html).
