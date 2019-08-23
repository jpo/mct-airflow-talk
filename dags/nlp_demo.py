import os
import pandas as pd
from datetime import datetime, timedelta
from string import punctuation

# airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# scikit-learn
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# spacy
import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS


class CleanTextTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return [t.strip().replace('\n', ' ').replace('\r', ' ').lower() for t in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


def tokenizer(text):
    parser = English()

    # Creating our token object, which is used to create documents with
    # linguistic annotations.
    tokens = parser(text)
    clean_tokens = []

    for token in tokens:
        if token.lemma_ != '-PRON-':
            t = token.lemma_.lower().strip()
        else:
            t = token.lower_

        if t not in STOP_WORDS and t not in punctuation:
            clean_tokens.append(t)


    # Lemmatizing each token and converting each token into lowercase
    # tokens = [t.lemma_.lower().strip() if t.lemma_ != "-PRON-" else t.lower_ for t in tokens]

    # Removing stop words
    # tokens = [t for t in tokens if t not in STOP_WORDS and t not in punctuation]

    # return preprocessed list of tokens
    return clean_tokens


# def print_most_informative(vectorizer, clf, N):
#     feature_names = vectorizer.get_feature_names()
#     coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
#     top_class1 = coefs_with_fns[:N]
#     top_class2 = coefs_with_fns[:-(N + 1):-1]

#     print('Class 1 best: ')
#     for feat in top_class1:
#         print(feat)

#     print('Class 2 best: ')
#     for feat in top_class2:
#         print(feat)


def load_training_data():
    pass


def clean_data():
    AIRFLOW_HOME = os.environ['AIRFLOW_HOME']
    csv_path = os.path.join(AIRFLOW_HOME, 'data', 'research_papers.csv')
    df = pd.read_csv(csv_path)
    print(df.head())
    print(df.isnull().sum())


def train_model():
    AIRFLOW_HOME = os.environ['AIRFLOW_HOME']
    csv_path = os.path.join(AIRFLOW_HOME, 'data', 'research_papers.csv')
    df = pd.read_csv(csv_path)
    train, test = train_test_split(df, test_size=0.33, random_state=42)

    pipeline = Pipeline([
        ('clean_text', CleanTextTransformer()),
        ('vectorizer', CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 1))),
        ('clf', LogisticRegression())
    ])

    # Train Model
    x_train = train['Title'].tolist()
    y_train = train['Conference'].tolist()
    pipeline.fit(x_train, y_train)

    # Test Model
    x_test = test['Title'].tolist()
    y_test = test['Conference'].tolist()

    y_pred = pipeline.predict(x_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def save_predictions():
    pass


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2018, 8, 1),
    "email": ["airflow@airflow.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG(
    "nlp_demo",
    default_args=default_args,
    schedule_interval='@once')

with dag:
    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data)

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model)

    dag >> clean_data_task >> train_model_task
