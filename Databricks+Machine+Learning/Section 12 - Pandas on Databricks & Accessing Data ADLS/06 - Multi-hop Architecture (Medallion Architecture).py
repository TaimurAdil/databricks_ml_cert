# Databricks notebook source
# MAGIC %md
# MAGIC **Multi-hop Architecture (Medallion Architecture):** https://www.databricks.com/glossary/medallion-architecture

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1: Mount 3 containers (bronze, silver, & gold)

# COMMAND ----------

# Application (client) ID
application_id = "acf24c54-2edc-411a-89ee-184ab0c2b79c"

# Directory (tenant) ID
directory_id = "aca956d5-1716-486c-9340-ffedc6d009ae"

# Secret value
secret_value = "EOW8Q~2yGX2ezRXF5lTqV-55ez9uEXyvq_roec-2"

# COMMAND ----------

container_name = "bronze"
account_name = "gen2storage05"
mount_point = "/mnt/bronze"

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

container_name = "silver"
account_name = "gen2storage05"
mount_point = "/mnt/silver"

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

container_name = "gold"
account_name = "gen2storage05"
mount_point = "/mnt/gold"

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

# MAGIC %md
# MAGIC # Part 2: Read data from bronze container

# COMMAND ----------

bank_data = spark.read.csv("dbfs:/mnt/bronze/bank_data.csv", header=True)

# COMMAND ----------

display(bank_data)

# COMMAND ----------

# Drop 3 columns (CustomerId, Surname, Gender)
bank_data = bank_data.drop("CustomerId", "Surname", "Gender")

# COMMAND ----------

display(bank_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3: Write data in silver layer

# COMMAND ----------

bank_data.write.parquet("/mnt/silver/bank_data")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 4: Write data in gold container

# COMMAND ----------

bank_data = spark.read.parquet("/mnt/silver/bank_data")

# COMMAND ----------

display(bank_data)

# COMMAND ----------

bank_data = bank_data[bank_data['Balance'] !=0]

# COMMAND ----------

display(bank_data)

# COMMAND ----------

bank_data.write.parquet("/mnt/gold/bank_data")

# COMMAND ----------

bank_data = spark.read.parquet("/mnt/gold/bank_data")

# COMMAND ----------

display(bank_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 5: Unmount the 3 containers

# COMMAND ----------

# unmount the storage container (bronze)
dbutils.fs.unmount('/mnt/bronze')

# COMMAND ----------

# unmount the storage container (silver)
dbutils.fs.unmount('/mnt/silver')

# COMMAND ----------

# unmount the storage container (gold)
dbutils.fs.unmount('/mnt/gold')

# COMMAND ----------

# Display the mounts
display(dbutils.fs.mounts())
