# Databricks notebook source
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

# COMMAND ----------

import pandas as pd
 
from pyspark.sql.functions import monotonically_increasing_id, expr, rand
import uuid

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

raw_data = spark.read.load("/databricks-datasets/wine-quality/winequality-red.csv",format="csv",sep=";",inferSchema="true",header="true" )


# COMMAND ----------

raw_data

# COMMAND ----------

display(raw_data)

# COMMAND ----------

def addIdColumn(dataframe, id_column_name):
    """Add id column to dataframe"""
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]
 
def renameColumns(df):
    """Rename columns to be compatible with Feature Store"""
    renamed_df = df
    for column in df.columns:
        renamed_df = renamed_df.withColumnRenamed(column, column.replace(' ', '_'))
    return renamed_df

# COMMAND ----------

renamed_df = renameColumns(raw_data)
df = addIdColumn(renamed_df, 'wine_id')

# COMMAND ----------

features_df = df.drop('quality')
display(features_df)

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS wine_db")
 
# Create a unique table name for each run. This prevents errors if you run the notebook multiple times
table_name = f"wine_db_" + str(uuid.uuid4())[:6]
print(table_name)

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

fs.create_table(
    name=table_name,
    primary_keys=["wine_id"],
    df=features_df,
    schema=features_df.schema,
    description="wine features"
)

# COMMAND ----------

inference_data_df = df.select("wine_id", "quality", (10 * rand()).alias("real_time_measurement"))
display(inference_data_df)

# COMMAND ----------

def load_data(table_name, lookup_key):
    # In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]
 
    # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="quality", exclude_columns="wine_id")
    training_pd = training_set.load_df().toPandas()
 
    # Create train and test datasets
    X = training_pd.drop("quality", axis=1)
    y = training_pd["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

# COMMAND ----------

X_train, X_test, y_train, y_test, training_set = load_data(table_name, "wine_id")
X_train.head()

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
 
client = MlflowClient()
 
try:
    client.delete_registered_model("wine_model") # Delete the model if already created
except:
    None

# COMMAND ----------

import mlflow

# COMMAND ----------

# Disable MLflow autologging and instead log the model using Feature Store
mlflow.sklearn.autolog(log_models=False)
 
def train_model(X_train, X_test, y_train, y_test, training_set, fs):
    ## fit and log model
    with mlflow.start_run() as run:
 
        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
 
        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))
 
        fs.log_model(
            model=rf,
            artifact_path="wine_quality_prediction",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name="wine_model",
        )
 
train_model(X_train, X_test, y_train, y_test, training_set, fs)

# COMMAND ----------

batch_input_df = inference_data_df.drop("quality") # Drop the label column
 
predictions_df = fs.score_batch("models:/wine_model/latest", batch_input_df) # Here we are using score_batch function of the Feature Store object 'fs' to make batch predictions on the model
                                  
display(predictions_df["wine_id", "prediction"])

# COMMAND ----------

so2_cols = ["free_sulfur_dioxide", "total_sulfur_dioxide"]

# COMMAND ----------

new_features_df = (features_df.withColumn("average_so2", expr("+".join(so2_cols)) / 2))

# COMMAND ----------

display(new_features_df)

# COMMAND ----------

fs.write_table(
    name=table_name,
    df=new_features_df,
    mode="merge"
)

# COMMAND ----------

display(fs.read_table(name=table_name))

# COMMAND ----------

def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]
    
    # fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="quality", exclude_columns="wine_id")
    training_pd = training_set.load_df().toPandas()
 
    # Create train and test datasets
    X = training_pd.drop("quality", axis=1)
    y = training_pd["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set
 
X_train, X_test, y_train, y_test, training_set = load_data(table_name, "wine_id")
X_train.head()

# COMMAND ----------

def train_model(X_train, X_test, y_train, y_test, training_set, fs):
    ## fit and log model
    with mlflow.start_run() as run:
 
        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
 
        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))
 
        fs.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name="wine_model",
        )
 
train_model(X_train, X_test, y_train, y_test, training_set, fs)

# COMMAND ----------

batch_input_df = inference_data_df.drop("quality") # Drop the label column
predictions_df = fs.score_batch(f"models:/wine_model/latest", batch_input_df)
display(predictions_df["wine_id","prediction"])

# COMMAND ----------


