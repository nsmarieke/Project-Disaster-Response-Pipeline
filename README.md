# Disaster Response Pipeline Project

### Summary of the project
In this project the disaster data from Appen is analyzed.
The data will be categorized such that the messages could be send to the appropriate disaster relief agency.
A web app will display visualizations of the data.


### Explanation of the files
There are two pipelines; the ETL pipeline that cleans data and stores it in a database and a ML pipeline that trains classifier and saves the results.
The ETL pipeline: Under data the process_data.py file can be found and this file loads, cleans and saves the data.
The data that is loaded are the disaster_categories.csv and disaster_messages.csv.
The data is saved in the data folder.
The ML pipeline: Under models the train_classifier.py can be found and this file loads and tokenizes the data than builds, trains, evaluate and saves the model.
The data that is loaded is the data saved in the ETL pipeline.


### Instructions to run the pipelines:
- To run ETL pipeline:
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline:
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

### Instructions to run the web app
1. Run the following command in the app's directory to run your web app:
    `python run.py`
2. Go to http://0.0.0.0:3001/
