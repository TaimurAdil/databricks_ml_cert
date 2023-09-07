# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC **Mounting cloud object storage:**
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/dbfs/mounts

# COMMAND ----------

# Code from azure documentation

configs = {"fs.azure.account.auth.type": "OAuth",
          "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
          "fs.azure.account.oauth2.client.id": "<application-id>",
          "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope="<scope-name>",key="<service-credential-key-name>"),
          "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/<directory-id>/oauth2/token"}

# Optionally, you can add <directory-name> to the source URI of your mount point.
dbutils.fs.mount(
  source = "abfss://<container-name>@<storage-account-name>.dfs.core.windows.net/",
  mount_point = "/mnt/<mount-name>",
  extra_configs = configs)

# COMMAND ----------

# Display the mounts
display(dbutils.fs.mounts())

# COMMAND ----------

# MAGIC %fs ls

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1: App registrations details

# COMMAND ----------

# Application (client) ID
application_id = "7bea0f66-ea6c-4543-95fd-71a6be09f955"

# Directory (tenant) ID
directory_id = "aca956d5-1716-486c-9340-ffedc6d009ae"

# Secret value
secret_value = "g968Q~lj-z2vvUfe0mhYn7DydZIvsURG.nfnGafr"

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: Service app permission 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3: Mounting ADLS to DBFS

# COMMAND ----------

# Application (client) ID
application_id = "7bea0f66-ea6c-4543-95fd-71a6be09f955"

# Directory (tenant) ID
directory_id = "aca956d5-1716-486c-9340-ffedc6d009ae"

# Secret value
secret_value = "g968Q~lj-z2vvUfe0mhYn7DydZIvsURG.nfnGafr"

# COMMAND ----------

# Code we will use

configs = {"fs.azure.account.auth.type": "OAuth",
          "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
          "fs.azure.account.oauth2.client.id": "7bea0f66-ea6c-4543-95fd-71a6be09f955",
          "fs.azure.account.oauth2.client.secret": "g968Q~lj-z2vvUfe0mhYn7DydZIvsURG.nfnGafr",
          "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/aca956d5-1716-486c-9340-ffedc6d009ae/oauth2/token"}

# Optionally, you can add <directory-name> to the source URI of your mount point.
dbutils.fs.mount(
  source = "abfss://input@gen2storage05.dfs.core.windows.net/",
  mount_point = "/mnt/input",
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

# MAGIC %md
# MAGIC # Part 4: Unmounting ADLS to DBFS

# COMMAND ----------

dbutils.fs.unmount("/mnt/input")

# COMMAND ----------

# Display the mounts
display(dbutils.fs.mounts())
