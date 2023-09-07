# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC **Mounting cloud object storage:**
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/dbfs/mounts

# COMMAND ----------

# Application (client) ID
application_id = "acf24c54-2edc-411a-89ee-184ab0c2b79c"

# Directory (tenant) ID
directory_id = "aca956d5-1716-486c-9340-ffedc6d009ae"

# Secret value
secret_value = "EOW8Q~2yGX2ezRXF5lTqV-55ez9uEXyvq_roec-2"

# COMMAND ----------

container_name = "input"
account_name = "gen2storage05"
mount_point = "/mnt/input"

# COMMAND ----------

# Code we will use

configs = {"fs.azure.account.auth.type": "OAuth",
          "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
          "fs.azure.account.oauth2.client.id": f"{application_id}",
          "fs.azure.account.oauth2.client.secret": f"{secret_value}",
          "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{directory_id}/oauth2/token"}

# Optionally, you can add <directory-name> to the source URI of your mount point.
dbutils.fs.mount(
  source = f"abfss://{container_name}@{account_name}.dfs.core.windows.net/",
  mount_point = mount_point,
  extra_configs = configs)

# COMMAND ----------

# Display the mounts
display(dbutils.fs.mounts())

# COMMAND ----------

# Reading data from new mount point

bank_data = spark.read.csv("dbfs:/mnt/input/bank_data.csv", header=True)

# COMMAND ----------

display(bank_data)

# COMMAND ----------

# Unmounting ADLS to DBFS
dbutils.fs.unmount("/mnt/input")

# COMMAND ----------

# Display the mounts
display(dbutils.fs.mounts())

# COMMAND ----------


