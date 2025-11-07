# Pytorch Training
```{index} double: Tutorial; PyTorch Training
```

```{raw} html
<span class="index-entry">Tutorial</span>
<span class="index-entry">PyTorch Training</span>
```

This tutorial demonstrates how to set up a FastIoT project that trains machine learning models using PyTorch.
It includes services for data storage in MongoDB, data processing, and model training with PyTorch.
The tutorial guides you through creating the necessary services, configuring the project, and running the services using Docker.
It covers the following steps:
1. **Install Required Python Dependencies**: Instructions for installing necessary Python packages.
2. **Implement Services**: Code snippets for the Mongo Database Service, Data Processing Service,
3. **PyTorch Training Service**: Code snippet for the PyTorch Training Service.
4. **Run the Services**: Instructions to run the services and verify their functionality.
5. **Verify Model Training**: Steps to check the trained models and training logs.

## Tutorial Video

```{raw} html
<video width="640" controls poster="https://rwth-aachen.sciebo.de/public.php/dav/files/xoKCpNqy3k25Jrj/Folie5.PNG">
  <source src="https://rwth-aachen.sciebo.de/s/ZiS3HxetQ6KmnwY/download/KIOptiPack-Pytorch-Training.mov" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

### Commands 

| Step                                | Description                                                     | Command(s)                                                                                   |
|-------------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| **Install dependencies**            | Install all required packages                                   | `pip install fastiot[dev] pymongo kioptipack-dataprocessing mlflow psutil torch torchvision` |
| **Create new project**              | Scaffold a new FastIoT project                                  | `fiot create new-project save_data`                                                          |
| **Create producer service**         | Add a simple producer module                                    | `fiot create new-service data_source`                                                        |
| **Create a Mongo Database Service** | Service to interact with MongoDB database                       | `fiot create new-service mongo_database`                                                     |
| **Create Data Processing Service**  | Service to process data using kioptipack-dataprocessing package | `fiot create new-service data_processing`                                                    |
| **Create Pytorch Training Service** | Service to train models using PyTorch                           | `fiot create new-service pytorch_training`                                                   |
| **Configure project**               | Generate or update FastIoT configuration files                  | `fiot config`                                                                                |
| **Start for testing**               | Start broker and enable running individual services in PyCharm  | `fiot config`<br>`fiot start --use-test-deployment`                                          |
| **Build Docker images**             | Build all services as Docker images                             | `fiot config`<br>`fiot build`                                                                |
| **Start Docker containers**         | Start all services and broker via Docker                        | `fiot start full`                                                                            |
| Optional: **Adjust PyCharm logs**   | Make PyCharm’s terminal output more readable                    | *Run → Edit Configuration Templates → Modify Options → Emulate terminal in output console*   |
