# Save Data

This Blueprint considers two services: A data producer that generates and sends data to the Fast IoT broker, and a data consumer that receives the data from the broker and saves it to a database.
It is a minimal example to demonstrate how to save data using Fast IoT.
Below you find the code snippets for both services.

```{prereq}
- A **FastioT Project** created via the FastIoT CLI. 
    - You can find more information on how to create a FastIoT Project in the [FastIoT Documentation](https://fastiot.readthedocs.io/en/latest/tutorials/part_1_getting_started/01_first_project_setup.html).
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
```{tip}
If you name the servervices **exactly** as shown in this Blueprint, you can simply copy and paste the code snippets into the respective service files created by the FastIoT CLI.
- Mongo Database Service: **MongoDatabaseService** (created by running `fiot create service mongo_database`)
- Data Source Service: **DataSourceService** (created by running `fiot create service data_source`)
```
---

## Dependency Installation

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


```{literalinclude} ../../../fast-iot-example-projects/fiot-save-data/src/fiot_save_data_services/mongo_database/mongo_database_service.py
:language: python
:linenos: true
```

---

## Data Source Service
This service generates and sends data to the FastIoT broker at regular intervals.
The data is sent on a specific subject that the Mongo Database Service listens to.

```{note}
You can create a new service in your FastIoT project via the FastIoT CLI: 
`fiot create service data_source`.
```

```{literalinclude} ../../../fast-iot-example-projects/fiot-save-data/src/fiot_save_data_services/data_source/data_source_service.py
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
If you named your project differently than `fiot-save-data`, the path to the service directories will differ accordingly.
```

```{tip}
If you are using pycharm, you can run the services directly from the IDE by clicking the green play button in the gutter next to the `run.py` files in the respective service directories.
```

To run the services do the following:
1. Start your MongoDB instance if it is not already running.
2. Start the FastIoT broker in your project directory inside a terminal:
   ```bash
   fiot start integration_test
   ```
3. Run the Mongo Database using the `run.py` python file. You can use the **green play button ▶️** in PyCharm or run the following command in your terminal:
   ```bash
   python src/fiot_save_data_services/mongo_database/run.py
   ```
4. Run the Data Source using the `run.py` python file. You can use the **green play button ▶️** in PyCharm or run the following command in your terminal:
   ```bash
   python src/fiot_save_data_services/data_source/run.py
   ```
   
```{tip}
Steps 1 to 4 can also be done in a single step using the FastIoT CLI command:
`fiot start full`. This command starts the FastIoT broker and all services in your project.
```

---

## Verifying Data Storage
To verify that the data is being stored correctly in MongoDB, you can use the MongoDB shell or a GUI tool like MongoDB Compass to check the contents of the specified collection in your database.
1. Open the MongoDB shell or MongoDB Compass.
2. Connect to your MongoDB instance. 
3. Navigate to the database and collection specified in the Mongo Database Service.

You should see the data entries being added to the collection as they are received from the Data Source Service.
Below you can find an example of how the data entries might look in the MongoDB collection:

![Blueprints](../_static/blueprint-sava-data-expected-mongo-state.png)


