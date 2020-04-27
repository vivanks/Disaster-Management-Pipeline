# DisasterResponsePipeline

Objective: The objective is to create a disaster response pipeline.

# Pipeline Explained

It has three components in total

1. ETL module: It takes data from disaster_categories.csv and disaster_messages.csv (dataset). 
   Code: Process_data.py - Reads in the data, cleans and stores it in a SQL database.
   
2. Model module: The Train_classifier.py file includes the code necessary to load data the sql database, transform it using natural language processing and machine language algorithm. Has GridSearchCV and can train and test the model 

3. Web App Module: The run.py has the user interface needed to predict and display results. The templates folder contains the html template

# Instruction to run
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ (Tip: after executing step 1 and 2)
