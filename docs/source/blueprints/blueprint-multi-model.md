# Multi Model System Blueprint

This set of blueprints showcases a system that trains multiple models using different services within a FastIoT project. The system consists of four main services.
In Particular a **LightGBM**, **Pytorch**, **Tensorflow** are included in this system.
The Setup has two options for data storage: **MongoDB** and **MariaDB**.
Those are essentially drop-in replacements for each other, so you can choose the one that fits your needs best.
Note, that for the MariaDB option, you need to set up a MariaDB instance. 
The same applies for MongoDB.

```{prereq}
- A **FastioT Project** created via the FastIoT CLI. You can use the project created in a previous bueprint or create a new one.
- **Doker** installed and running on your machine (for example by installing [Docker Desktop](https://www.docker.com/products/docker-desktop/)).
- **MongoDB** as a database to save the data.
  - Make sure you have a **running** MongoDB instance. 
  - You can check if MongoDB is running by executing `mongod --version` in your terminal.
- **MariaDB** as a database to save the data.
  - Make sure you have a **running** MariaDB instance. 
  - You can check if MariaDB is running by executing `mariadb --version` in your terminal.
- install the required dependencies for the services. See the "Dependency Installation" section below.
```

```{tip}
If you name the servervices **exactly** as shown in this Blueprint, you can simply copy and paste the code snippets into the respective service files created by the FastIoT CLI.
- **DataProcessingService** (created by running `fiot create new-service data_processing`)
- **DatabaseMariaService** (created by running `fiot create new-service database_maria`)
- **DatabaseMongoService** (created by running `fiot create new-service database_mongo`)
- **DummyDataGenerationService** (created by running `fiot create new-service dummy_data_generation`)
- **MlConsumerService** (created by running `fiot create new-service ml_consumer`)
- **MlLightgbmRegressionMlflowService** (created by running `fiot create new-service ml_lightgbm_regression_mlflow`)
- **MlMonitoringService** (created by running `fiot create new-service ml_monitoring`)
- **MlPytorchRegressionService** (created by running `fiot create new-service ml_pytorch_regression`)
- **MlPytorchRegressionMlflowService** (created by running `fiot create new-service ml_pytorch_regression_mlflow`)
- **MlServingService** (created by running `fiot create new-service ml_serving`)
- **MlServingMlflowService** (created by running `fiot create new-service ml_serving_mlflow`)
- **MlTensorflowRegressionMlflowService** (created by running `fiot create new-service ml_tensorflow_regression_mlflow`)
```

---

## Description

This set of Blueprints consists of four main services:
1. **Data Processing Service**: This service periodically fetches raw data from the Database Service via the Database Service, processes it using a predefined data processing pipeline, and stores the processed data back in the database.
2. **Database Services**: These services connect to either a MongoDB or MariaDB database
3. **Dummy Data Generation Service**: This service generates dummy data and sends it to the Database Service via the FastIoT broker.
4. **ML Consumer Service**: This service retrieves the processed data from the Database Service and trains multiple models using different frameworks (LightGBM, Pytorch, Tensorflow). The training process, results, and model weights are logged to an mlflow server.
5. **ML Monitoring Service**: This service monitors the performance of the trained models and logs system metrics to the mlflow server.
6. **ML Serving Services**: These services serve the trained models via REST API endpoints, allowing for easy integration with other applications.
7. **ML LightGBM Regression MLflow Service**: This service trains a LightGBM regression model using the processed data and logs the training process, results, and model weights to an mlflow server.
8. **ML Pytorch Regression Service**: This service trains a Pytorch regression model using the processed data and logs the training process, results, and model weights to an mlflow server.
9. **ML Pytorch Regression MLflow Service**: This service trains a Pytorch regression model using the processed data and logs the training process, results, and model weights to an mlflow server.
10. **ML Tensorflow Regression MLflow Service**: This service trains a Tensorflow regression model using the processed data and logs the training process, results, and model weights to an mlflow server.
11. **ML Serving MLflow Service**: This service serves the trained models via REST API endpoints, allowing for easy integration with other applications.
12. **ML Serving Service**: This service serves the trained models via REST API endpoints, allowing for easy integration with other applications.

---

## Dependency Installation

The following Blueprint requires the `pymongo` package to connect to the MongoDB database.
You can install it via pip:
```bash
pip install pymongo
```
For MariaDB the `mariadb` package is required.
You can install it via pip:
```bash
pip install mariadb
```
The Blueprint also requires the `kioptipack-dataprocessing` for simplified broker communication.
You can install it via pip:
```bash
pip install kioptipack-dataprocessing
```

For experiment tracking the `mlflow` package is required.
You can install it via pip:
```bash
pip install mlflow
```
For the Weights and Biases integration the `wandb` package is required.
You can install it via pip:
```bash
pip install wandb
```
For stracking system metrics the `psutil` package is required.
You can install it via pip:
```bash
pip install psutil
```
Finally, the `lightgbm`, `torch` and `tensorflow` packages are required for
model training.
You can install them via pip:
```bash
pip install lightgbm
pip install torch torchvision
pip install tensorflow
```

---

### Data Processing Service

This service periodically fetches raw data from the Database Service, runs the processing pipeline and stores processed data back to the database.

```{literalinclude} ../../../fast-iot-example-projects/fiot-complex-system/src/complex_system_services/data_processing/data_processing_service.py
:language: python
:linenos: true
```

---

### Database MariaDB Service

```{literalinclude} ../../../fast-iot-example-projects/fiot-complex-system/src/complex_system_services/database_maria/database_maria_service.py
:language: python
:linenos: true
```

---

### Database MongoDB Service

```{literalinclude} ../../../fast-iot-example-projects/fiot-complex-system/src/complex_system_services/database_mongo/database_mongo_service.py
:language: python
:linenos: true
```

---

### Dummy Data Generation Service

```{literalinclude} ../../../fast-iot-example-projects/fiot-complex-system/src/complex_system_services/dummy_data_generation/dummy_data_generation_service.py
:language: python
:linenos: true
```

---

### ML Consumer Service

```{literalinclude} ../../../fast-iot-example-projects/fiot-complex-system/src/complex_system_services/ml_consumer/ml_consumer_service.py
:language: python
:linenos: true
```

---

### ML LightGBM Regression MLflow Service

```{literalinclude} ../../../fast-iot-example-projects/fiot-complex-system/src/complex_system_services/ml_lightgbm_regression_mlflow/ml_lightgbm_regression_mlflow_service.py
:language: python
:linenos: true
```

---

### ML Pytorch Regression Service

```{literalinclude} ../../../fast-iot-example-projects/fiot-complex-system/src/complex_system_services/ml_pytorch_regression/ml_pytorch_regression_service.py
:language: python
:linenos: true
```

---

### ML Pytorch Regression MLflow Service

```{literalinclude} ../../../fast-iot-example-projects/fiot-complex-system/src/complex_system_services/ml_pytorch_regression_mlflow/ml_pytorch_regression_mlflow_service.py
:language: python
:linenos: true
```

---

### ML Tensorflow Regression MLflow Service

```{literalinclude} ../../../fast-iot-example-projects/fiot-complex-system/src/complex_system_services/ml_tensorflow_regression_mlflow/ml_tensorflow_regression_mlflow_service.py
:language: python
:linenos: true
```

---

### ML Monitoring Service

```{literalinclude} ../../../fast-iot-example-projects/fiot-complex-system/src/complex_system_services/ml_monitoring/ml_monitoring_service.py
:language: python
:linenos: true
```

---

### ML Serving Service

```{literalinclude} ../../../fast-iot-example-projects/fiot-complex-system/src/complex_system_services/ml_serving/ml_serving_service.py
:language: python
:linenos: true
```

---

### ML Serving MLflow Service

```{literalinclude} ../../../fast-iot-example-projects/fiot-complex-system/src/complex_system_services/ml_serving_mlflow/ml_serving_mlflow_service.py
:language: python
:linenos: true
```

---

