# Databricks notebook source
# MAGIC %md
# MAGIC # Import required packages and load dataset

# COMMAND ----------

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

# If you are running Databricks Runtime for Machine Learning, `mlflow` is already installed and you can skip the following line. 
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC - The Iris dataset is a widely used dataset in the field of machine learning and data analysis. It is named after the Iris flower plant and was introduced by the British statistician and biologist Ronald Fisher in 1936. The dataset is frequently used as a beginner's dataset to learn and practice various classification algorithms.
# MAGIC
# MAGIC - The Iris dataset consists of measurements of four features of three different species of Iris flowers: Setosa, Versicolor, and Virginica
# MAGIC
# MAGIC - **The four features are:**
# MAGIC 1) Sepal length (in centimeters)
# MAGIC 2) Sepal width (in centimeters)
# MAGIC 3) Petal length (in centimeters)
# MAGIC 4) Petal width (in centimeters)
# MAGIC
# MAGIC - In the Iris dataset, the target variable is the species of the Iris flowers. It represents the class or category to which each sample belongs. The target variable is a categorical variable with three possible values: Setosa, Versicolor, and Virginica
# MAGIC
# MAGIC - For each of the three species, there are 50 samples, resulting in a total of 150 samples in the dataset

# COMMAND ----------

# Load the iris dataset from scikit-learn
iris = load_iris()
X = iris.data
y = iris.target

# COMMAND ----------

print(X)

# COMMAND ----------

print(y)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1. Single-machine Hyperopt workflow
# MAGIC
# MAGIC Here are the steps in a Hyperopt workflow:  
# MAGIC 1. Define a function to minimize.  
# MAGIC 2. Define a search space over hyperparameters.  
# MAGIC 3. Select a search algorithm.  
# MAGIC 4. Run the tuning algorithm with Hyperopt `fmin()`.
# MAGIC
# MAGIC For more information, see the [Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define a function to minimize
# MAGIC We use a support vector machine classifier. The objective is to find the best value for the regularization parameter `C`.  

# COMMAND ----------

def objective(C):
    # Create a support vector classifier model
    clf = SVC(C=C)
    
    # Use the cross-validation accuracy to compare the models' performance
    accuracy = cross_val_score(clf, X, y).mean()
    
    # Hyperopt tries to minimize the objective function. A higher accuracy value means a better model, so you must return the negative accuracy.
    return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the search space over hyperparameters

# COMMAND ----------

search_space = hp.lognormal('C', 0, 1.0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select a search algorithm
# MAGIC
# MAGIC The two main choices are:
# MAGIC * `hyperopt.tpe.suggest`: Tree of Parzen Estimators, a Bayesian approach which iteratively and adaptively selects new hyperparameter settings to explore based on past results
# MAGIC * `hyperopt.rand.suggest`: Random search, a non-adaptive approach that samples over the search space

# COMMAND ----------

algo = tpe.suggest

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the tuning algorithm with Hyperopt fmin()

# COMMAND ----------

argmin = fmin(
  fn=objective,
  space=search_space,
  algo=algo,
  max_evals=16
  )

# COMMAND ----------

print("Best value found: ", argmin)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2. Distributed tuning using Apache Spark and MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC - **Distributed tuning** A technique for tuning the hyperparameters of ML models on large datasets. It works by distributing the tuning process across multiple machines, which can significantly speed up the tuning process
# MAGIC
# MAGIC - Apache Spark is a distributed computing framework that can be used to run distributed tuning jobs
# MAGIC - MLflow is an open-source platform for managing the end-to-end ML lifecycle, including hyperparameter tuning

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC To distribute tuning, add one more argument to `fmin()`: Argument `Trials` & class `SparkTrials`
# MAGIC
# MAGIC `SparkTrials` takes 2 optional arguments:  
# MAGIC * `parallelism`: Number of models to fit and evaluate concurrently. The default is the number of available Spark task slots.
# MAGIC * `timeout`: Maximum time (in seconds) that `fmin()` can run. The default is no maximum time limit.

# COMMAND ----------

from hyperopt import SparkTrials

# COMMAND ----------

spark_trials = SparkTrials()

# COMMAND ----------

with mlflow.start_run():
  argmin = fmin(
    fn=objective,
    space=search_space,
    algo=algo,
    max_evals=16,
    trials=spark_trials
  )

# COMMAND ----------

print("Best value found: ", argmin)
