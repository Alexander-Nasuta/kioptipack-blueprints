# Project Initialization Tutorial
```{index} triple: FastIoT; Project Setup; Tutorial
```

```{raw} html
<span class="index-entry">FastIoT</span>
<span class="index-entry">Project Setup</span>
<span class="index-entry">Tutorial</span>
```

In this guide, you’ll learn how to set up a new FastIoT project inside a pre-existing PyCharm project, create a simple producer service, and start both the FastIoT broker and your service for local testing.
By the end of this tutorial, you will have:
- Installed required Python packages
- Created a FastIoT project structure
- Added and configured a custom service
- Started the FastIoT broker and tested inter-module communication

```{raw} html
<video width="640" controls poster="https://raw.githubusercontent.com/Alexander-Nasuta/Alexander-Nasuta/main/readme_images/logo.png">
  <source src="https://rwth-aachen.sciebo.de/s/J4GcJaRW8s6g5AA/download/KIOptiPack-Project-Setup-Tutorial.mov" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

## Prerequisites
Before starting, please ensure the following software is installed and working:

| Tool               | Description                             | Check Command               |
| ------------------ | --------------------------------------- | --------------------------- |
| **PyCharm**        | Your Python IDE                         | `Help → About`              |
| **Docker**         | Required for service containers         | `docker --version`          |
| **Docker Desktop** | (Optional) for GUI container management | *System Tray → Docker Icon* |
| **Conda**          | Environment manager for Python          | `conda --version`           |


### Commands 

| Step                              | Description                                                    | Command(s)                                                                                 |
|-----------------------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **Install dependencies**          | Install all required packages                                  | `pip install fastiot[dev]`                                                                 |
| **Create new project**            | Scaffold a new FastIoT project                                 | `fiot create new-project TestTutorial`                                                     |
| **Create producer service**       | Add a simple producer module                                   | `fiot create new-service producer`                                                         |
| **Configure project**             | Generate or update FastIoT configuration files                 | `fiot config`                                                                              |
| **Start for testing**             | Start broker and enable running individual services in PyCharm | `fiot config`<br>`fiot start --use-test-deployment`                                        |
| **Build Docker images**           | Build all services as Docker images                            | `fiot config`<br>`fiot build`                                                              |
| **Start Docker containers**       | Start all services and broker via Docker                       | `fiot start full`                                                                          |
| Optional: **Adjust PyCharm logs** | Make PyCharm’s terminal output more readable                   | *Run → Edit Configuration Templates → Modify Options → Emulate terminal in output console* |
