# Databricks notebook source
from pyspark.sql.types import DoubleType, StringType, StructType, StructField
 
schema = StructType([
  StructField("longitude", DoubleType(), True),
  StructField("latitude", DoubleType(), True),
  StructField("housing_median_age", DoubleType(), True),
  StructField("total_rooms", DoubleType(), True),
  StructField("total_bedrooms", DoubleType(), True),
  StructField("population", DoubleType(), True),
  StructField("households", DoubleType(), True),
  StructField("median_income", DoubleType(), True),
  StructField("median_house_value", DoubleType(), True),
  StructField("ocean_proximity", StringType(), True)
])
 
housing_df = spark.read.format("csv").schema(schema).option("header", "true").load("/FileStore/housing.csv")

# COMMAND ----------

housing_df.display()

# COMMAND ----------

housing_df.count()

# COMMAND ----------

housing_df.columns

# COMMAND ----------

housing_df.dtypes

# COMMAND ----------

type(housing_df)

# COMMAND ----------

train_df, test_df = housing_df.randomSplit([0.99, 0.01], seed=42)

# COMMAND ----------

from databricks import automl

# COMMAND ----------

summary = automl.regress(train_df, target_col="median_house_value", timeout_minutes=5)

# COMMAND ----------

print(summary)

# COMMAND ----------

print(summary.best_trial.model_path)

# COMMAND ----------

import mlflow
 
model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"
 
predict = mlflow.pyfunc.spark_udf(spark, model_uri)
pred_df = test_df.withColumn("prediction", predict(*test_df.drop("median_house_value").columns))
display(pred_df)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
 
regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="median_house_value", metricName="r2")
rmse = regression_evaluator.evaluate(pred_df)
print(f"val_r2_score on test dataset: {rmse:.3f}")

# COMMAND ----------


