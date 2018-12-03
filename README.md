# Disaster Response Pipeline Project

In this project, the dataset consists of messages that were sent by people during disaster events. I used this dataset to build a model that classifies these messages , so that these messages could be then forwarded to the appropriate relief agency.
It includes a web app(using Flask Framework) where an emergency worker can input a new message and get classification results in various categories. The web app also displays visualisations (using Plotly) of the data.

This project showcases my data engineering skills including ETL and ML Pipeline preparation, which makes use of model through a web app API and data visualisaton .

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db` 
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
