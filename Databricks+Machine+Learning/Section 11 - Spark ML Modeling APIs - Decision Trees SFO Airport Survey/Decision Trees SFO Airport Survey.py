# Databricks notebook source
# MAGIC %md 
# MAGIC # Part 1: Business Problem (SFO Survey)

# COMMAND ----------

# MAGIC %md
# MAGIC - Each year, San Francisco Airport (SFO) conducts a customer satisfaction survey to find out what they are doing well and where they can improve
# MAGIC - The survey gauges satisfaction with SFO facilities, services, and amenities. SFO compares results to previous surveys to discover elements of the guest experience that are not satisfactory
# MAGIC - The 2013 SFO Survey Results consists of customer responses to survey questions and an overall satisfaction rating with the airport
# MAGIC - Whether we could use machine learning to predict a customer's overall response given their responses to the individual questions
# MAGIC - You may think this is not very useful because the customer has already provided an overall rating as well as individual ratings for various aspects of the airport such as parking, food quality and restroom cleanliness. However, we didn't stop at prediction instead we asked the question: 
# MAGIC
# MAGIC **What factors drove the customer to give the overall rating?**
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC **Outline of our data flow:**  
# MAGIC 1) Load the dataset
# MAGIC 2) Understand the data: Compute statistics and create visualizations to get a better understanding of the data
# MAGIC 3) Create Model
# MAGIC 4) Evaluate the model
# MAGIC 5) Feature Importance: Determine the importance of each of the individual ratings in determining the overall rating by the customer

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: Load the Data

# COMMAND ----------

survey = spark.read.csv("dbfs:/databricks-datasets/sfo_customer_survey/2013_SFO_Customer_Survey.csv", header="true", inferSchema="true")

# COMMAND ----------

display(survey)

# COMMAND ----------

len(survey.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3: Understand the Data

# COMMAND ----------

survey.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC - As you can see above there are many questions in the survey including what airline the customer flew on, where do they live, etc. For the purposes of answering the above, we will focus on the Q7A, Q7B, Q7C .. Q7O questions since they directly related to customer satisfaction, which is what we want to measure
# MAGIC
# MAGIC - The possible values for the above are:  
# MAGIC
# MAGIC 0 = no answer, 1 = Unacceptable, 2 = Below Average, 3 = Average, 4 = Good, 5 = Outstanding, 6 = Not visited or not applicable
# MAGIC
# MAGIC Select only the fields you are interested in

# COMMAND ----------

dataset = survey.select("Q7A_ART", "Q7B_FOOD", "Q7C_SHOPS", "Q7D_SIGNS", "Q7E_WALK", "Q7F_SCREENS", "Q7G_INFOARR", "Q7H_INFODEP", "Q7I_WIFI", "Q7J_ROAD", "Q7K_PARK", "Q7L_AIRTRAIN", "Q7M_LTPARK", "Q7N_RENTAL", "Q7O_WHOLE")

# COMMAND ----------

len(dataset.columns)

# COMMAND ----------

# Let's start with the overall rating
from pyspark.sql.functions import *
dataset.selectExpr('avg(Q7O_WHOLE) Q7O_WHOLE').take(1)

# COMMAND ----------

# The overall rating is only 3.87, so slightly above average. Let's get the averages of the constituent ratings

avgs = dataset.selectExpr('avg(Q7A_ART) Q7A_ART', 'avg(Q7B_FOOD) Q7B_FOOD', 'avg(Q7C_SHOPS) Q7C_SHOPS', 'avg(Q7D_SIGNS) Q7D_SIGNS', 'avg(Q7E_WALK) Q7E_WALK', 'avg(Q7F_SCREENS) Q7F_SCREENS', 'avg(Q7G_INFOARR) Q7G_INFOARR', 'avg(Q7H_INFODEP) Q7H_INFODEP', 'avg(Q7I_WIFI) Q7I_WIFI', 'avg(Q7J_ROAD) Q7J_ROAD', 'avg(Q7K_PARK) Q7K_PARK', 'avg(Q7L_AIRTRAIN) Q7L_AIRTRAIN', 'avg(Q7M_LTPARK) Q7M_LTPARK', 'avg(Q7N_RENTAL) Q7N_RENTAL')
display(avgs)

# COMMAND ----------

display(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 4: Create Model

# COMMAND ----------

# MAGIC %md So basic statistics can't seem to answer the question: **What factors drove the customer to give the overall rating?**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Replace responce 0 & 6 with average rating

# COMMAND ----------

# MAGIC %md
# MAGIC - To treat responses, **0 = No Answer** and **6 = Not Visited or Not Applicable** as missing values
# MAGIC - **First option:** Replace missing values with column mean
# MAGIC - **Second option:** Set all the values of 0 or 6 to the average rating of 3

# COMMAND ----------

training = dataset.withColumn("label", dataset['Q7O_WHOLE']*1.0).na.replace(0,3).replace(6,3)

# COMMAND ----------

display(training)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the ML pipeline

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# Defining the input columns
inputCols = ['Q7A_ART', 'Q7B_FOOD', 'Q7C_SHOPS', 'Q7D_SIGNS', 'Q7E_WALK', 'Q7F_SCREENS', 'Q7G_INFOARR', 'Q7H_INFODEP', 'Q7I_WIFI', 'Q7J_ROAD', 'Q7K_PARK', 'Q7L_AIRTRAIN', 'Q7M_LTPARK', 'Q7N_RENTAL']

# Creating a VectorAssembler
va = VectorAssembler(inputCols=inputCols,outputCol="features")

# Creating a DecisionTreeRegressor
dt = DecisionTreeRegressor(labelCol="label", featuresCol="features", maxDepth=4)

# Creating a RegressionEvaluator
evaluator = RegressionEvaluator(metricName = "rmse", labelCol="label")

# Creating a ParamGridBuilder 
grid = ParamGridBuilder().addGrid(dt.maxDepth, [3, 5, 7, 10]).build()

# Creating a CrossValidator
cv = CrossValidator(estimator=dt, estimatorParamMaps=grid, evaluator=evaluator, numFolds = 10)

# Creating a Pipeline
pipeline = Pipeline(stages=[va, dt])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the Model

# COMMAND ----------

model = pipeline.fit(training)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View the tree

# COMMAND ----------

display(model.stages[-1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make predictions

# COMMAND ----------

predictions = model.transform(training)
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 5: Evaluate the model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate the model using root mean squared error (RMSE)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator()

evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the model

# COMMAND ----------

import uuid
model_save_path = f"/tmp/sfo_survey_model/{str(uuid.uuid4())}"
model.write().overwrite().save(model_save_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 6: Feature Importance

# COMMAND ----------

# feature importances
model.stages[1].featureImportances

# COMMAND ----------

# map the features to their proper names to make them easier to read

featureImportance = model.stages[1].featureImportances.toArray()
featureNames = map(lambda s: s.name, dataset.schema.fields)
featureImportanceMap = zip(featureImportance, featureNames)

# COMMAND ----------

print(featureImportanceMap)

# COMMAND ----------

# Convert featureImportanceMap to a Dataframe

importancesDf = spark.createDataFrame(sc.parallelize(featureImportanceMap).map(lambda r: [r[1], float(r[0])]))

# COMMAND ----------

display(importancesDf)

# COMMAND ----------

# Rename column names
importancesDf = importancesDf.withColumnRenamed("_1", "Feature").withColumnRenamed("_2", "Importance")

# COMMAND ----------

display(importancesDf)

# COMMAND ----------

# visulization of the Feature Importances (pie chart)
display(importancesDf.orderBy(desc("Importance")))

# COMMAND ----------

# visulization of the Feature Importances (bar chart)
display(importancesDf.orderBy(desc("Importance")))

# COMMAND ----------

importancesDf.orderBy(desc("Importance")).limit(3).agg(sum("Importance")).take(1)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The 3 most important features are:
# MAGIC
# MAGIC 1. Signs 
# MAGIC 2. Screens
# MAGIC 3. Food
# MAGIC
# MAGIC These 3 features combine to make up 88% of the overall rating

# COMMAND ----------

# delete saved model
dbutils.fs.rm(model_save_path, True)
