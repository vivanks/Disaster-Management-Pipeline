import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('stopwords')
import re
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    
    '''
    Parameters: 
        database_filepath - file path of sqlite database
    
    Returns:
       X - message data
       Y - categories
       category_names- labels for 36 categories
    '''
        
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    
    return X, Y, category_names


def tokenize(text):
    
    '''
    Parameters: 
        text - message text
    Returns:
       lemmed - cleaned text
    '''
    
    
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    lemmed = [lemmatizer.lemmatize(w, pos='v').strip() for w in lemmed]
    
    return lemmed

def build_model():
    
    '''
    Build a machine learning pipeline using count-vectorizer, Tf-Idf, and Random forest
    
    Returns
      Results of GridSearchCV
    '''
    
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    
    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Evaluate model performance
    
    Parameters
      model - model to be evaluated
      X_test - test messages
      Y_test - true categories of the test messages
      category_name - labels for 36 categories
    
    Returns
     print accuracy and classification report for each category
    '''
    
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    
    '''
    Save the model as pickle file
    
    Parameters
     model - model to be saved
     model_filepath - path of the pickle file
   
   Returns
     model saved as pickle format in the model_filepath destination    
    '''
    
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()