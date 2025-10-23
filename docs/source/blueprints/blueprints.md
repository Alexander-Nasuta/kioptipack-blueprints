# Blueprints

![Blueprints](../_static/Blueprint-concept.png)

Within the **KIOptiPack** project, the IoT platform [FastIoT](https://github.com/FraunhoferIVV/fastiot) is developed and used as a core component of the software technology stack.  
This repository contains so-called **Blueprints**: reusable code templates that demonstrate how to implement **Machine Learning (ML)** use cases with the FastIoT framework.  

The goal of the Blueprints is to provide developers with a solid starting point for rapidly building and deploying ML workflows using FastIoT.

FastIoT follows a **microservice architecture** and uses [NATS](https://nats.io/) as a message broker to enable communication between services.  
This architecture introduces specific design considerations for implementing ML use cases.


The Blueprints in this repository are designed to address these challenges. 
They provide ready-to-use templates that streamline the development of ML services in FastIoT, including components for **data preprocessing**, **model training**, **storage**, and **model serving**.

```{toctree}
:maxdepth: 2

blueprint-save-data
blueprint-data-processing-pipeline
blueprint-pytorch-training
blueprint-lightgbm-training
blueprint-tensorflow-training
blueprint-apheris-training
blueprint-model-serving
```
