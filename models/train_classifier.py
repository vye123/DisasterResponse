# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC


# load data from database
def load_data(database_filepath):
    """
	    Loads data from SQL Database table
	
	    Input:
	    database_filepath: SQL database file
	
	    Output:
	    X pandas_dataframe: Features dataframe
	    Y pandas_dataframe: Target dataframe
	    category_names list: Target labels 
	    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse_table', con = engine)
    X = df['message']
    Y = df.iloc[:,4:]
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    
    """
	    Tokenizes text data
	
	    Input:
	    Messages as text data
	
	    Output:
	    Processed text after normalizing, tokenizing and lemmatizing
	    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:

        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
        
    """
    Build a Trained Model with GridSearchCV

    """
    # Set pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
                            OneVsRestClassifier(LinearSVC())
        )
        )
    ])

    # Set parameters for gird search
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # Set grid search
    model = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, verbose=3)


    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Evaluate the model, print classification report and accuracy score
    
    Input:
    model: The trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    
    Output:
    Classification report
    Accurary score
    
    """

    y_pred = model.predict(X_test)
    
    # print classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))
    
def save_model(model, model_filepath):
    
    """
    Save the model to a file
    
    Input:
        model: The trained model with GridSearchCV
        model_filepath: The filepath where the pickel file will be saved at.
        
    Output:
        The model file
        
    """
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