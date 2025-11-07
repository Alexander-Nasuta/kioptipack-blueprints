# Save Data Blueprint Tutorial
```{index} double: Save Data; Tutorial
```

```{raw} html
<span class="index-entry">Save Data</span>
<span class="index-entry">Tutorial</span>
```
This guide walks you through the Save Data Blueprint, which demonstrates how to set up a data processing pipeline using FastIoT services. The pipeline fetches raw data from a MongoDB database, processes it using the `kioptipack-dataprocessing` package, and stores the processed data back into the database.
The guid follows these steps:
1. **Set Up MongoDB Database**: A setup guide for MongoDB on your local machine. An UI for easier database management is also introduced.
2. **Install Required Python Dependencies**: Instructions for installing necessary Python packages.
3. **Implement Services**: Code snippets for the Mongo Database Service and Data Processing Service.
4. **Run the Services**: Instructions to run the services and verify their functionality.
5. **Verify Data Processing**: Steps to check the processed data in the MongoDB database.

```{raw} html
<video width="640" controls poster="https://rwth-aachen.sciebo.de/public.php/dav/files/xoKCpNqy3k25Jrj/Folie3.PNG">
  <source src="https://rwth-aachen.sciebo.de/s/YXLzKzG9MxKFQqN/download/KIOPtipack-Sava-Data_Blueprint-Tutorial.mov" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

### Commands 

| Step                                | Description                                                    | Command(s)                                                                                 |
|-------------------------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **Install dependencies**            | Install all required packages                                  | `pip install fastiot[dev] pymongo kioptipack-dataprocessing`                                                        |
| **Create new project**              | Scaffold a new FastIoT project                                 | `fiot create new-project save_data`                                                        |
| **Create producer service**         | Add a simple producer module                                   | `fiot create new-service data_source`                                                      |
| **Create a Mongo Database Service** | Service to interact with MongoDB database                      | `fiot create new-service mongo_database`                                                   |
| **Configure project**               | Generate or update FastIoT configuration files                 | `fiot config`                                                                              |
| **Start for testing**               | Start broker and enable running individual services in PyCharm | `fiot config`<br>`fiot start --use-test-deployment`                                        |
| **Build Docker images**             | Build all services as Docker images                            | `fiot config`<br>`fiot build`                                                              |
| **Start Docker containers**         | Start all services and broker via Docker                       | `fiot start full`                                                                          |
| Optional: **Adjust PyCharm logs**   | Make PyCharm’s terminal output more readable                   | *Run → Edit Configuration Templates → Modify Options → Emulate terminal in output console* |
