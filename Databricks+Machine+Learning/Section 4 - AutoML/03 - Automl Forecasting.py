# Databricks notebook source
import pyspark.pandas as ps
df = ps.read_csv("/databricks-datasets/COVID/covid-19-data")
df["date"] = ps.to_datetime(df['date'], errors='coerce')
df["cases"] = df["cases"].astype(int)
display(df)

# COMMAND ----------

df.count()

# COMMAND ----------

df.columns

# COMMAND ----------

df.describe()

# COMMAND ----------

import databricks.automl
import logging

# COMMAND ----------

logging.getLogger("py4j").setLevel(logging.WARNING)

# COMMAND ----------

help(databricks.automl.forecast)

# COMMAND ----------

summary = databricks.automl.forecast(df, target_col='cases', time_col='date', horizon=30, frequency="d", primary_metric='mdape', output_database="default", timeout_minutes=15)

# COMMAND ----------

print(summary)

# COMMAND ----------

print(summary.output_table_name)

# COMMAND ----------

forecast_pd = spark.table(summary.output_table_name)
display(forecast_pd)

# COMMAND ----------

import mlflow.pyfunc
from mlflow.tracking import MlflowClient
 
run_id = MlflowClient()
trial_id = summary.best_trial.mlflow_run_id
 
model_uri = "runs:/{run_id}/model".format(run_id=trial_id)
pyfunc_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

forecasts = pyfunc_model._model_impl.python_model.predict_timeseries()
display(forecasts)

# COMMAND ----------

df_true = df.groupby("date").agg(y=("cases", "avg")).reset_index().to_pandas()
import matplotlib.pyplot as plt
 
fig = plt.figure(facecolor='w', figsize=(10, 6))
ax = fig.add_subplot(111)
forecasts = pyfunc_model._model_impl.python_model.predict_timeseries(include_history=True)
fcst_t = forecasts['ds'].dt.to_pydatetime()
ax.plot(df_true['date'].dt.to_pydatetime(), df_true['y'], 'k.', label='Observed data points')
ax.plot(fcst_t, forecasts['yhat'], ls='-', c='#0072B2', label='Forecasts')
ax.fill_between(fcst_t, forecasts['yhat_lower'], forecasts['yhat_upper'],
                color='#0072B2', alpha=0.2, label='Uncertainty interval')
ax.legend()
plt.show()


# COMMAND ----------


