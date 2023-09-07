# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC **Resources:**
# MAGIC
# MAGIC **1) Connect to Azure Data Lake Storage Gen2 and Blob Storage -** https://learn.microsoft.com/en-us/azure/databricks/external-data/azure-storage#access-azure-data-lake-storage-gen2-or-blob-storage-using-a-sas-token
# MAGIC
# MAGIC **2) Azure Data Lake Storage Gen2 URI -** https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-abfs-driver

# COMMAND ----------

# Code from azure documentation

spark.conf.set("fs.azure.account.auth.type.<storage-account>.dfs.core.windows.net", "SAS")
spark.conf.set("fs.azure.sas.token.provider.type.<storage-account>.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
spark.conf.set("fs.azure.sas.fixed.token.<storage-account>.dfs.core.windows.net", dbutils.secrets.get(scope="<scope>", key="<sas-token-key>"))

# COMMAND ----------

# Code we will use

spark.conf.set("fs.azure.account.auth.type.gen2storage05.dfs.core.windows.net", "SAS")
spark.conf.set("fs.azure.sas.token.provider.type.gen2storage05.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
spark.conf.set("fs.azure.sas.fixed.token.gen2storage05.dfs.core.windows.net", "sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupyx&se=2023-08-08T19:28:57Z&st=2023-08-08T11:28:57Z&spr=https&sig=YrVUaMFJXzPjNc9TJd%2BFnfuP6fzvmj5XqqiC0wDKYUw%3D")

# COMMAND ----------

bank_data = spark.read.csv("abfss://input@gen2storage05.dfs.core.windows.net/bank_data.csv", header=True)

# COMMAND ----------

display(bank_data)
