import sys
import pickle
import pandas as pd
import numpy as np

import nltk

nltk.download(['punkt', 'wordnet'])

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = 'disasterResponse'
    df = pd.read_sql_table(table_name, con=engine)
    X = df.message.values
    y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    return X, y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__n_estimators': [100, 250]
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2, cv=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_preds = model.predict(X_test)
    print(classification_report(Y_test, y_preds, target_names=category_names))


def save_model(model, model_filepath):
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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()