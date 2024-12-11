# Federated Learning Control System
This directory is the control system for the Federated Learning (FL) environment.  The control system is responsible for managing the FL training process.  It will start the server and client containers, and monitor the training process.  The app directory contains the web-app for the control system.

## Table of Contents
- [Federated Learning Control System](#federated-learning-control-system)
  - [Table of Contents](#table-of-contents)
  - [Directory Structure](#directory-structure)

## Directory Structure
    ├── README.md                       <- The top-level README for developers using this project.
    ├── dockerfile                      <- Dockerfile for building the control system container
    ├── dockerfile-compose.yml          <- Docker-compose file for building the control system container
    └── app                             <- Directory for the control system web-app
        ├──app.py                       <- Main application file
        ├──templates                    <- Directory for html templates
        │  ├──index.html                <- Main page template
        │  ├──client.html               <- Client page template
        │  └──server.html               <- Server page template
        └──static                       <- Directory for static files
            ├──style.css                <- CSS file for styling the web-app
            ├──script.js                <- JavaScript file for scripting
            ├──containers.js            <- JavaScript file for tab management
            └──*.png                    <- Image files for the web-app

## Setup Instructions
The control system is intended to be started via the start_control.sh script.  This script will build the control system container and start the container.  The control system will be accessible via a web browser at http://localhost:5000. 

```bash
../start_control.sh
```
During initialization, the control system will ask for the raw data directory.  This directory will be mounted as a volume into the control container.  The raw data directory should contain the raw data files for the FL training process.  The control system will use this data to create the training data for the client nodes.

## Files
### App.py
The control system will run the app.py file on startup.  This app defines the routes for the webpage, while also providing set actions for webpage interactions with the host system.  All actions are predefined in this file, and are triggered by webpage interactions.  There are three main functions provided from the webpage:
- Start Server: This function will start the containers for the server-side.  This includes the SuperLink and SuperExec containers
- Start Client: This function will start the containers for the client-side.  This includes the SuperNode and ClientApp containers
- Preprocess Data: This function will parse the provided raw data directory, and create the required input data for the model at each client. None of this data is transmitted over the internet.

### Index.html
The index.html file is the main page for the control system.  This file outlines the overall structure of the webpage, and provides an area for containers to be loaded into the webpage. By default, the index.html file will load the client.html file into the container section. Tabs on the header of the webpage will allow the user to switch between the client and server pages.  The server page is currently locked behing a password, and is not accessible to general clients.

### Client.html
The client.html file contains the necessary containers for clients interacting with the system.  It provides access to request data preprocessing, as well as start the client containers.  The client.html file is the default page loaded into the control system. The terminal on this page will display the output from the client containers. This page will also display the current status of the system, including: GPU status, data preprocessing status, and client container status.

### Server.html
The server.html file contains the necessary containers for the server interacting with the system.  It provides access to start the server containers.  The server.html file is currently locked behind a password, and is not accessible to general clients.  The terminal on this page will display the output from the server containers. This page will also display a useful visualization of all connected nodes.

### Script.js
The script.js file contains the necessary JavaScript for the webpage.  This file will handle all webpage interactions, and will trigger the appropriate actions in the app.py file.  This file will also handle the loading of the client and server pages into the webpage.

### Style.css
The style.css file contains the necessary CSS for the webpage.  This file will handle all styling for the webpage, and will provide a consistent look and feel for the webpage.

## Dependencies
All dependencies for this system are installed in the provided docker container, and it is recommended to run the control system in the provided container.

