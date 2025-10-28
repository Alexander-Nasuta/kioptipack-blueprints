# Data Processing Pipeline
The Data Processing Pipeline Example extends the **Save Data Blueprints** and introduces a data processing service that preprocesses the data before it is used for model training or inference.
It Fetches raw data from the database, processes it using a predefined pipeline, and stores the processed data back in the database.
Storing the data in the database is done via the Database Service. 
That way the Database Service is the only service that interacts with the database.
This abstraction allows to easily change the database implementation (for example for replacing it with another database) without changing the other services.

```{prereq}
- A **FastioT Project** created via the FastIoT CLI. You can use the project created in the Save Data Blueprint example or create a new one.
- **Doker** installed and running on your machine (for example by installing [Docker Desktop](https://www.docker.com/products/docker-desktop/)).
- **MongoDB** as a database to save the data.
  - Make sure you have a **running** MongoDB instance. 
  - You can check if MongoDB is running by executing `mongod --version` in your terminal. 
- install the required dependencies for the services. See the "Dependency Installation" section below.
```

```{tutorial}
You Can find a Video Tutorial on how to create a FastIoT Project and add Services in the tutorials section of the documentation.
```

```{note}
The imports from `kio` namespace are part of the `kioptipack-dataprocessing` package, which provides simplified communication with the FastIoT broker.
```

```{note}
The Blueprints in this example build upon the Blueprints from the Save Data Blueprint example. The Database Service has more functionality to support storing processed data.
So make sure to not forget to add the additional functionality to the Database Service when starting from the Save Data Blueprint example.
```

```{tip}
If you name the servervices **exactly** as shown in this Blueprint, you can simply copy and paste the code snippets into the respective service files created by the FastIoT CLI.
- Mongo Database Service: **MongoDatabaseService** (created by running `fiot create service mongo_database`)
- Data Source Service: **DataSourceService** (created by running `fiot create service data_source`)
- Data Processing Service: **DataProcessingService** (created by running `fiot create service data_processing`)
```

---

## Description

This set of Blueprints consists of three main services:
1. **Data Source Service**: This service features a dataset, that will be saved to the MongoDB database via the FastIoT broker.
2. **Mongo Database Service**: This service connects to a MongoDB database and listens for incoming data messages from the FastIoT broker. Upon receiving data, it saves the raw data to a specified collection in the MongoDB database.
3. **Data Processing Service**: This service periodically fetches raw data from the MongoDB database via the Database Service, processes it using a predefined data processing pipeline, and stores the processed data back in the database.

The raw datapoints of this example have the following structure:
```python
{
        "ListeKomponenten": ["K000055", "K000057"], # List of materials (id or material name)
        "Massenanteile": [0.5, 0.5], # mass ratios of the materials (unit: g/g)
        "Flächenanteilmodifiziert": 0, # modified surface (unit: %)
        "Geometrie": "Quader", # geometry (unit: list of types)
        "Kopfraumatmosphäre": None, # headspace atmosphere (unit: Pa)
        "Masse": None, # mass (unit: g)
        "Verpackungstyp": "Folie", # packaging type
        "CAD": None, # link to CAD file
        "RauheitRa": 0.08666666666666667, # roughness Ra (unit: µm)
        "RauheitRz": 0.924, # roughness Rz (unit: µm)
        "Trübung": 216.1, # haze (unit: HLog)
        "Glanz": 36.7, # gloss (unit: GE)
        "Dicke": 738.6666666666666, # thickness (unit: µm)
        "Emodul": 807.9225728004443, # elastic modulus (unit: MPa)
        "MaximaleZugspannung": 33.22942107172407, # maximum tensile stress (unit: MPa)
        "MaximaleLängenänderung": 14.57795412214027, # maximum elongation (unit: %)
        "Ausformung": 3, # forming process rating  (unit: class (1 to 6))
        "Kaltverfo": 3, # cold forming rating (unit: class (1 to 3))
        "Temp": 420, # [LABEL] temperature (unit: °C) 
        "Zeit": 32, # [LABEL] time (unit: s)
        "Druck": 1, # [LABEL] pressure (unit: bar)
}
```
The dataset contains labeled datapoints related to material validation processes for thermoforming equipment. 
Each datapoint reflects a specific use case or validation scenario, including information on the material used, equipment configuration, validation method, and outcome.

Thermoforming machine manufacturers typically offer a predefined material portfolio aligned with their equipment specifications. 
Converters can select suitable materials based on the intended application. 
To ensure contractual performance targets are met, manufacturers validate the machine's efficiency using standard materials. 
Using alternative materials is generally at the converter's own risk. 
However, some manufacturers support formal validation of unapproved materials—either through practical testing on identical or comparable machines or via structured methods such as Design of Experiments (DOE).
Each datapoint captures the relevant parameters and results from these validation efforts.

The Data Processing Pipeline transforms the raw datapoints, such that they can be used for model training or inference.
For Machine Learning applications ideally all features should be numerical values in the range of 0 to 1 or sometimes -1 to 1.
The Python Package *kioptipack-dataprocessing* provides various preprocessing steps that can be used to transform the raw data into a suitable format for Machine Learning applications.
It is based on Scikit-Learn and Pandas.
The Data Processing Service creates a pandas DataFrame from the raw data and applies Pipeline to preprocess the data.
Such a pipeline can consist of various preprocessing steps, such as:
- Filling missing values
- Encoding categorical variables
- Scaling numerical features
- Feature engineering

The processed data is then stored back in the MongoDB database via the Database Service.
In this example, the processed data is stored in a separate collection named *KIOptiPackProcessed*.
Below can find an example datapoint after processing:

```python
{
    "M_K000034": 0.0, # mass ratio of material K000034
    "M_K000035": 0.0, # mass ratio of material K000035
    "M_K000055": 0.5, # mass ratio of material K000055
    "M_K000057": 0.5, # mass ratio of material K000057
    "M_K000141": 0.0, # mass ratio of material K000141
    "Temp": 0.6, # [LABEL] scale to 0.0 - 1.0
    "Zeit": 0.2, # [LABEL] scale to 0.0 - 1.0
    "Druck": 0.7216, # [LABEL] scale to 0.0 - 1.0
}
```
In this example we excluded a lot of fields to keep the blueprint concise. 

```{tip}
Check out the [**Open Hub Days Demo Project**](https://github.com/Alexander-Nasuta/openhub-demo) for a more comprehensive implementation of a data processing pipeline.
In that project a more sophisticated data processing pipeline is implemented.
```

```{note}
A scientific publication describing the data processing pipeline and the Machine Learning models for the example dataset is arleady accepted and will be referenced here once published.
```

---

## Dependency Installation
```{note}
If you have already installed the dependencies in the Save Data Blueprint example, you can skip this section.
```

The following Blueprint requires the `pymongo` package to connect to the MongoDB database.
You can install it via pip:
```bash
pip install pymongo
```
The Blueprint also requires the `kioptipack-dataprocessing` for simplified broker communication.
You can install it via pip:
```bash
pip install kioptipack-dataprocessing
```

---

## Data Source Service
```{index} single: Dataset;
```
```{raw} html
<span class="index-entry">Dataset</span>
```

The Data Source Service holds a dataset and sends the data to the FastIoT broker to be saved in the MongoDB database via the Database Service.
In this example the dataset is loaded from the `kioptipack-dataprocessing` package.
In your own implementation you can replace this with loading your own dataset from a CSV file or any other source.

```{literalinclude} ../../../fast-iot-example-projects/fiot-data-processing-pipeline/src/fiot_data_processing_pipeline_services/data_source/data_source_service.py
:language: python
:linenos: true
```

---

## Mongo Database Service
```{index} single: Database 
```
```{index} single: MongoDB
```

```{raw} html
<span class="index-entry">Database</span>
<span class="index-entry">MongoDB</span>
```
This service connects to a MongoDB database and saves incoming data to a specified collection.
It listens for messages on a specific subject from the FastIoT broker and inserts the received data into the MongoDB collection.

```{note}
You can create a new service in your FastIoT project via the FastIoT CLI: 
`fiot create service mongo_database`.
```

```{note}
The Blueprint uses the **defaulf MongoDB connection** parameters. If you left the default settings while setting up your MongoDB instance, you do not need to change anything.
**If you do not use the default settings**, you need to change the following values when starting the service:
 - `_db_port`: the port your MongoDB instance is running on (default: `27017`)
 - `_db_host`: the host your MongoDB instance is running on (default: `localhost`)
```

```{literalinclude} ../../../fast-iot-example-projects/fiot-data-processing-pipeline/src/fiot_data_processing_pipeline_services/mongo_database/mongo_database_service.py
:language: python
:linenos: true
```
---
## Data Processing Service

```{index} single: Dataprocessing
```

```{raw} html
<span class="index-entry">Dataprocessing</span>
```

This service periodically fetches raw data from the MongoDB database via the Database Service, processes it using a predefined data processing pipeline, and stores the processed data back in the database.


```{literalinclude} ../../../fast-iot-example-projects/fiot-data-processing-pipeline/src/fiot_data_processing_pipeline_services/data_processing/data_processing_service.py
:language: python
:linenos: true
```

---

## Running the Services

```{note}
Make sure that you have run `fiot config` at least once in your project directory to set up the FastIoT broker configuration.
For more information, refer to the [FastIoT Documentation](https://fastiot.readthedocs.io/en/latest/tutorials/part_1_getting_started/02_fiot_config.html).
```

```{note}
If you named your project differently than `fiot-data-processing-pipeline`, the path to the service directories will differ accordingly.
```

```{tip}
If you are using pycharm, you can run the services directly from the IDE by clicking the green play button in the gutter next to the `run.py` files in the respective service directories.
```

To run the services do the following:
1. Start your **MongoDB** instance if it is not already running.
2. Start the **FastIoT broker** in your project directory inside a terminal:
   ```bash
   fiot start integration_test
   ```
3. Run the **Mongo Database Service** using the `run.py` python file. You can use the **green play button ▶️** in PyCharm or run the following command in your terminal:
   ```bash
   python src/fiot_data_processing_pipeline/mongo_database/run.py
   ```
4. Run the **Data Source Service** using the `run.py` python file. You can use the **green play button ▶️** in PyCharm or run the following command in your terminal:
   ```bash
   python src/fiot_data_processing_pipeline/data_source/run.py
   ```
5. Run the **Data Processing Service** using the `run.py` python file. You can use the **green play button ▶️** in PyCharm or run the following command in your terminal:
   ```bash
   python src/fiot_data_processing_pipeline/data_source/run.py
   ```
   
---
## Verifying Data Processing
To verify that the data is being stored correctly in MongoDB, you can use the MongoDB shell or a GUI tool like MongoDB Compass to check the contents of the specified collection in your database.
1. Open the MongoDB shell or MongoDB Compass.
2. Connect to your MongoDB instance. 
3. Navigate to the database and collection specified in the Mongo Database Service (by default it is **KIOptiPackProcessed**).

You should see the data entries being added to the collection as they are received from the Data Source Service.
Below you can find an example of how the data entries might look in the MongoDB collection:

![Blueprints](../_static/blueprint-dataprocessing-pipeline-expected-mongo-screenshot.png)

