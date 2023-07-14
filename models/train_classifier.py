import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'omw-1.4'])
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

def load_data(database_filepath):
    '''
    Load data from database and split in X and Y 

    args:
        - database_filepath: path to the data
    
    returns:
        - X: input variable
        - Y: output variables
        - category_names: names of the output variables
    '''
    # Load data
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('data/DisasterResponse.db', engine)

    # X and Y and category names
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    Normalize, tokenize and lemmatize the given text

    Args:
        - text: The input text

    Returns:
        - lemmed: processed words from the input text
    '''
    # Normalize data
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # Tokenize text by word
    words = word_tokenize(text)
    
    # Lemmatize the words
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed


def build_model():
    '''
    Builds pipeline

    Args:
        None
    
    Returns:
        pipeline
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])

    parameters = {
        'clf__estimator__max_depth': (10, 20, 30),
        'clf__estimator__n_estimators': (10, 20, 30),
        'clf__estimator__min_samples_split': [2, 4, 6]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates model
    
    Args:
        model: pipeline
        X_test: test input variable
        Y_test: test output variable
        category_names: names of the categories
    
    Returns:
        None
    '''
    y_pred = model.predict(X_test)
    classification_report(Y_test, y_pred, target_names=category_names)


def save_model(model, model_filepath):
    '''
    Saves model
    
    Args:
        model: model
        model_filepath: path where the model is saved

    Returns:
        None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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