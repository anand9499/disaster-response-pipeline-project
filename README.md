# Disaster Response Pipeline Project

### Project Motivation
In this project, we will build a model to classify messages that were sent during disasters. There are 36 pre-defined categories such as Aid Related, Medical Help, Search and Rescue etc. By classifying these messages, we can help direct these messages to appropriate disaster relief agency. This project involves building of a basic ETL and ML Pipeline to facilitate this task. This is also a multi-label classification task , since a message can belong to one or more categories.

Finally, this project contains a web app, built using Flask framework, where we can input a message and get classification results.

### Files in repo
    -  app
        - template
            - master.html  # main page of web app
            - go.html  # classification result page of web app
        - run.py  # Flask file that runs app

    - data
        - disaster_categories.csv  # data to process
        - disaster_messages.csv  # data to process
        - process_data.py
        - DisasterResponse.db   # database to save clean data to

    - models
       - train_classifier.py


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db` 
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![webapp_screenshot](https://user-images.githubusercontent.com/16004232/53821796-73155b80-3f94-11e9-9149-7b8b1812c9b1.JPG)

![webapp_screenshot1](https://user-images.githubusercontent.com/16004232/53821813-7f99b400-3f94-11e9-87ec-54368c69ca45.JPG)

### Acknowledgements
We will be using the dataset provided by [Figure Eight](https://www.figure-eight.com/) that contains messages sent during disaster events.
