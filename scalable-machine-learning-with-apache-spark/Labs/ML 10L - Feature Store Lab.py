# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-c7e3123c-9ace-4d9a-89a1-10307b238964
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Feature Store Lab
# MAGIC
# MAGIC Now that you are familiar with the <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html" target="_blank">Databricks Feature Store</a>, try applying the concepts we learned to a new dataset below.
# MAGIC
# MAGIC The Feature Store Python API documentation can be found <a href="https://docs.databricks.com/dev-tools/api/python/latest/index.html#feature-store-python-api-reference" target="_blank">here</a>.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC
# MAGIC By the end of this lab, you should be able to;
# MAGIC
# MAGIC * Create a feature store and save features to this store
# MAGIC * Update an existing feature store table
# MAGIC * Register a MLflow model with feature store tables
# MAGIC * Perform batch scoring with a feature store

# COMMAND ----------

# DBTITLE 0,--i18n-0d9e386c-d2a5-4390-9a68-783da2b6eaeb
# MAGIC %md 
# MAGIC ## Lab Setup
# MAGIC
# MAGIC The first thing we're going to do is to **run setup script**. This script will define the required configuration variables that are scoped to each user.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 0,--i18n-2861b87d-095f-44c7-848e-9e4ea539ed2d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Load the Dataset
# MAGIC For this example, we will use a new COVID-19 dataset. Run the cell below to create our dataframe **`covid_df`**.

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

file_path = f"{DA.paths.datasets}/COVID/coronavirusdataset/Time.csv"
covid_df = (spark.read
            .format("csv")
            .option("header",True)
            .option("inferSchema", True)
            .load(file_path)
            .withColumn("index", monotonically_increasing_id()))

display(covid_df)

# COMMAND ----------

# DBTITLE 0,--i18n-e0cab81f-4d2b-486b-9af7-01f6fb212168
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Run the cell below to set up a database and unique table name **`table_name`** for the lab.

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DA.schema_name}")
table_name = f"{DA.schema_name}.coronavirus"

print(f"Table Name: {table_name}")

# COMMAND ----------

# DBTITLE 0,--i18n-9db69933-359c-46df-88df-523c20aec03e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's set up our FeatureStoreClient **`fs`**. 
# MAGIC
# MAGIC To create a feature store client, initialize a **`FeatureStoreClient`** object from the **`feature_store`** module.

# COMMAND ----------

# TODO
from databricks import feature_store
 
fs = # FILL_IN

# COMMAND ----------

# DBTITLE 0,--i18n-961b8810-5fce-4b34-b066-75f33a12e4f8
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Extract Features
# MAGIC
# MAGIC In this simple example we want to predict the number of daily deceased using the other information from from day. 
# MAGIC
# MAGIC Before we write to our feature table, we will need to write a feature computation function that separates our features from the label. 
# MAGIC
# MAGIC Fill in the feature computation function below to select only the feature columns, not **`deceased`**.

# COMMAND ----------

# TODO
@feature_store.feature_table
def select_features(dataframe):
    return # FILL_IN

covid_features_df = select_features(covid_df)
display(covid_features_df)

# COMMAND ----------

# DBTITLE 0,--i18n-d3892ba8-77ca-403f-869f-a6e0b0706555
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Create Feature Table
# MAGIC
# MAGIC Now that we have our features ready, complete the cell below to create our feature table.
# MAGIC
# MAGIC Make sure to set the name to the **`table_name`** we defined above.
# MAGIC
# MAGIC **NOTE:** The primary key needs to be defined in a list as follows: ["primary key name"]

# COMMAND ----------

# TODO
fs.create_table(
    name=#FILL_IN,
    primary_keys=#FILL_IN,
    df=#FILL_IN,
    schema=#FILL_IN,
    description=#FILL_IN
)

# COMMAND ----------

# DBTITLE 0,--i18n-a7a6fb53-a77d-465b-b9bd-1671e3afbbd6
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Update Feature Table
# MAGIC
# MAGIC Imagine now that we wanted to add separate columns for the month and day of the date for each entry. 
# MAGIC
# MAGIC Rather than recompute the table with these values, we just want to append these new columns to the existing table. 
# MAGIC
# MAGIC First, let's create columns for the month and day.

# COMMAND ----------

from pyspark.sql.functions import month, dayofmonth

add_df = (covid_features_df
  .select("date", "index")
  .withColumn("month", month("date"))
  .withColumn("day", dayofmonth("date"))
)

display(add_df)

# COMMAND ----------

# DBTITLE 0,--i18n-c040ac8a-f24d-4951-82bc-af1f509a90c2
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now we want to add this information to our feature table using **`write_table`**. 
# MAGIC
# MAGIC **NOTE:** Remember, we can use either **`"overwrite"`** or **`"merge"`** mode. Which one should we use here?

# COMMAND ----------

# TODO
fs.write_table(
    name=#FILL_IN,
    df=#FILL_IN,
    mode=#FILL_IN
)

# COMMAND ----------

# DBTITLE 0,--i18n-67d3eb45-0083-4a4c-a391-92ce31628b42
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now try using **`fs.read_table`**, specifying the **`table_name`** to see our updated feature table.

# COMMAND ----------

# TODO
updated_df = #FILL_IN

display(updated_df)

# COMMAND ----------

# DBTITLE 0,--i18n-529e3acd-adc0-45e3-b2a7-eed04a3911b2
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Training 
# MAGIC
# MAGIC Now that we have our feature table, we are ready to use it for model training. We'll need our target variable **`deceased`** in addition to our features, so let's get that first.

# COMMAND ----------

target_df = covid_df.select(["index", "deceased"])

display(target_df)

# COMMAND ----------

# DBTITLE 0,--i18n-386a9f38-3c89-41ba-9ce2-3645aa727411
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now let's create our training and test datasets.

# COMMAND ----------

from sklearn.model_selection import train_test_split

def load_data(table_name, lookup_key):
    model_feature_lookups = [feature_store.FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
    training_set = fs.create_training_set(target_df, model_feature_lookups, label="deceased", exclude_columns=["index","date"])
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("deceased", axis=1)
    y = training_pd["deceased"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

X_train, X_test, y_train, y_test, training_set = load_data(table_name, "index")
X_train.head()

# COMMAND ----------

# DBTITLE 0,--i18n-3d1f0e1f-5ead-4f3d-81d5-d400231d0e43
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now we can train a model and register it to the feature store.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()

suffix = DA.unique_name("-")
model_name = f"feature-store-covid_{suffix}"

try:
    client.delete_registered_model(model_name) # Deleting model if already created
except:
    None

# COMMAND ----------

import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

def train_model(table_name):
    X_train, X_test, y_train, y_test, training_set = load_data(table_name, "index")

    ## fit and log model
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("r2", r2_score(y_test, y_pred))

        fs.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=model_name,
            input_example=X_train[:5],
            signature=infer_signature(X_train, y_train)
        )
    
train_model(table_name)

# COMMAND ----------

# DBTITLE 0,--i18n-1836dc20-0e77-467e-af8f-33dcfdac7209
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now we have a trained model! Check the Feature Store UI to see that our model is now there. Can you tell which features this model uses from that table and which we excluded?
# MAGIC
# MAGIC Finally, let's apply the model.

# COMMAND ----------

## For sake of simplicity, we will just predict on the same inference_data_df
batch_input_df = target_df.drop("deceased") # Exclude true label
predictions_df = fs.score_batch(f"models:/{model_name}/1", 
                                  batch_input_df, result_type="double")
display(predictions_df)

# COMMAND ----------

# DBTITLE 0,--i18n-a2c7fb12-fd0b-493f-be4f-793d0a61695b
# MAGIC %md 
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
