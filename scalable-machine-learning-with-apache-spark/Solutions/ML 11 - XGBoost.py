# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-b9944704-a562-44e0-8ef6-8639f11312ca
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # XGBoost
# MAGIC
# MAGIC Up until this point, we have only used SparkML. Let's look a third party library for Gradient Boosted Trees. 
# MAGIC  
# MAGIC Ensure that you are using the <a href="https://docs.microsoft.com/en-us/azure/databricks/runtime/mlruntime" target="_blank">Databricks Runtime for ML</a> because that has Distributed XGBoost already installed. 
# MAGIC
# MAGIC **Question**: How do gradient boosted trees differ from random forests? Which parts can be parallelized?
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC
# MAGIC * Build a XGBoost model and integrate it into Spark ML pipeline
# MAGIC * Evaluate XGBoost model performance
# MAGIC * Compare and contrast most common gradient boosted approaches

# COMMAND ----------

# DBTITLE 0,--i18n-1e2c921e-1125-4df3-b914-d74bf7a73ab5
# MAGIC %md 
# MAGIC ## 📌 Requirements
# MAGIC
# MAGIC **Required Databricks Runtime Version:** 
# MAGIC * Please note that in order to run this notebook, you must use one of the following Databricks Runtime(s): **12.2.x-cpu-ml-scala2.12**

# COMMAND ----------

# DBTITLE 0,--i18n-6a1bb996-7b50-4f03-9bcd-3d3ec3069a6d
# MAGIC %md 
# MAGIC ## Lesson Setup
# MAGIC
# MAGIC The first thing we're going to do is to **run setup script**. This script will define the required configuration variables that are scoped to each user.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 0,--i18n-3e08ca45-9a00-4c6a-ac38-169c7e87d9e4
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Data Preparation
# MAGIC
# MAGIC Let's go ahead and index all of our categorical features, and set our label to be **`log(price)`**.

# COMMAND ----------

from pyspark.sql.functions import log, col
from pyspark.ml.feature import StringIndexer, VectorAssembler

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.withColumn("label", log(col("price"))).randomSplit([.8, .2], seed=42)

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price") & (field != "label"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

# DBTITLE 0,--i18n-733cd880-143d-42c2-9f29-602e48f60efe
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Pyspark Distributed XGBoost
# MAGIC
# MAGIC Let's create our distributed XGBoost model. While technically not part of MLlib, you can integrate <a href="https://databricks.github.io/spark-deep-learning/_modules/sparkdl/xgboost/xgboost.html" target="_blank">XGBoost</a> into your ML Pipelines. 
# MAGIC
# MAGIC To use the distributed version of Pyspark XGBoost you can specify two additional parameters:
# MAGIC
# MAGIC * **`num_workers`**: The number of workers to distribute over. Requires MLR 9.0+.
# MAGIC * **`use_gpu`**: Enable to utilize GPU based training for faster performance (optional).
# MAGIC
# MAGIC **NOTE:** **`use_gpu`** requires an ML GPU runtime. Currently, at most one GPU per worker will be used when doing distributed training.

# COMMAND ----------

from sparkdl.xgboost import XgboostRegressor
from pyspark.ml import Pipeline

params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": 42, "missing": 0}

xgboost = XgboostRegressor(**params)

pipeline = Pipeline(stages=[string_indexer, vec_assembler, xgboost])
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# DBTITLE 0,--i18n-8d5f8c24-ee0b-476e-a250-95ce2d73dd28
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Evaluate Model Performance
# MAGIC
# MAGIC Now we can evaluate how well our XGBoost model performed. Don't forget to exponentiate!

# COMMAND ----------

from pyspark.sql.functions import exp, col

log_pred_df = pipeline_model.transform(test_df)

exp_xgboost_df = log_pred_df.withColumn("prediction", exp(col("prediction")))

display(exp_xgboost_df.select("price", "prediction"))

# COMMAND ----------

# DBTITLE 0,--i18n-364402e1-8073-4b24-8e03-c7e2566f94d2
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Compute some metrics.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(exp_xgboost_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(exp_xgboost_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# DBTITLE 0,--i18n-21cf0d1b-c7a8-43c0-8eea-7677bb0d7847
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Alternative Gradient Boosted Approaches
# MAGIC
# MAGIC There are lots of other gradient boosted approaches, such as <a href="https://catboost.ai/" target="_blank">CatBoost</a>, <a href="https://github.com/microsoft/LightGBM" target="_blank">LightGBM</a>, vanilla gradient boosted trees in <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.GBTClassifier.html?highlight=gbt#pyspark.ml.classification.GBTClassifier" target="_blank">SparkML</a>/<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html" target="_blank">scikit-learn</a>, etc. Each of these has their respective <a href="https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db" target="_blank">pros and cons</a> that you can read more about.

# COMMAND ----------

# DBTITLE 0,--i18n-a2c7fb12-fd0b-493f-be4f-793d0a61695b
# MAGIC %md 
# MAGIC
# MAGIC ## Classroom Cleanup
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson:

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
