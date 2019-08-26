import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

# airflow
from airflow import DAG
from airflow.hooks.sqlite_hook import SqliteHook
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import BranchPythonOperator, PythonOperator

# scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


AIRFLOW_HOME = os.environ['AIRFLOW_HOME']
CLASSIFIERS = [LogisticRegression, LinearSVC, MultinomialNB]
# CLASSIFIERS = [LogisticRegression, LinearSVC, MultinomialNB, SGDClassifier]
# CLASSIFIERS = [LogisticRegression, LinearSVC, MultinomialNB, SGDClassifier, RandomForestClassifier]


def load_csv_to_sqlite(csv_path, sqlite_table, **context):
    """
    Loads CSV data (i.e. research_papers and mct_talks) in the default Sqlite DB.
    """
    df = pd.read_csv(csv_path)
    with SqliteHook().get_conn() as conn:
        df.to_sql(sqlite_table, con=conn, index=False, if_exists='replace')


def train_test_split(**context):
    """
    Splits the sample data (i.e. research_papers) into training and test sets
    and stores them in the Sqlite DB.
    """
    with SqliteHook().get_conn() as conn:
        # Load full dataset
        df = pd.read_sql('select * from research_papers', con=conn)

        # Create train/test split
        train = df.sample(frac=0.67, random_state=42)
        test = df.drop(train.index)

        # Save training and test data in separate tables
        train.to_sql('training_data', con=conn, index=False, if_exists='replace')
        test.to_sql('test_data', con=conn, index=False, if_exists='replace')


def train_model(classifier, **context):
    """
    Trains a model using the given classifier and stores it at MODEL_PATH.
    """
    # Load data
    with SqliteHook().get_conn() as conn:
        df = pd.read_sql('select * from training_data', con=conn)

    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(stop_words='english')),
        ('classifier', classifier())
    ])

    # Train model
    pipeline.fit(df['Title'].tolist(), df['Conference'].tolist())

    # Save model
    model_path = os.path.join(AIRFLOW_HOME, 'models', f'{classifier.__name__}.pkl')
    pickle.dump(pipeline, open(model_path, 'wb'))


def test_model(classifier, **context):
    """
    Tests a model created by the given classifier and returns its accuracy score
    as an XCom.
    """

    # Simulate a task failure
    # raise RuntimeError('BOOM!')

    # Load model
    model_path = os.path.join(AIRFLOW_HOME, 'models', f'{classifier.__name__}.pkl')
    pipeline = pickle.load(open(model_path, 'rb'))

    # Load data
    with SqliteHook().get_conn() as conn:
        df = pd.read_sql('select * from test_data', con=conn)

    # Make predictions
    X, y = df['Title'].tolist(), df['Conference'].tolist()
    y_pred = pipeline.predict(X)

    # Log classification results
    print(classification_report(y, y_pred))

    # Push Accuracy as XCom
    return classifier.__name__, f1_score(y, y_pred, average='weighted')


def predict(classifier, **context):
    """
    Makes predictions for a model created by the given classifier and returns its
    stores the results in the mct_talks table.
    """
    # Load model
    model_path = os.path.join(AIRFLOW_HOME, 'models', f'{classifier.__name__}.pkl')
    pipeline = pickle.load(open(model_path, 'rb'))

    with SqliteHook().get_conn() as conn:
        # Load data
        df = pd.read_sql('select * from mct_talks', con=conn)

        # Make predictions
        X = df['Title'].tolist()
        df['Conference'] = pipeline.predict(X)

        # Save predictions
        df.to_sql('mct_talks', con=conn, index=False, if_exists='replace')


def select_best_model(**context):
    """
    Selects the best model for classifying MCT Talks and selects the next
    appropriate task.
    """
    # Get Task IDs for test tasks
    task_ids = [f'test_{clf.__name__}' for clf in CLASSIFIERS]

    # Pull XCom from test tasks
    values = context['ti'].xcom_pull(key='return_value', task_ids=task_ids)

    # Find the classifier with the best accuracy
    classifier, _ = values[np.argmax(values, axis=0)[1]]

    # Return the task for the best classifier
    return f'predict_{classifier}'


# DAG Setup
default_args = {
    "owner": "airflow",
    "start_date": datetime(2018, 8, 1),
    "depends_on_past": False,
    "email": ["airflow@airflow.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "provide_context": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    'nlp_demo',
    default_args=default_args,
    schedule_interval='@once')

with dag:
    load_research_papers_task = PythonOperator(
        task_id='load_reasearch_papers',
        python_callable=load_csv_to_sqlite,
        op_kwargs={
            'csv_path': os.path.join(AIRFLOW_HOME, 'data', 'research_papers.csv'),
            'sqlite_table': 'research_papers'
        })

    load_mct_talks_task = PythonOperator(
        task_id='load_mct_talks',
        python_callable=load_csv_to_sqlite,
        op_kwargs={
            'csv_path': os.path.join(AIRFLOW_HOME, 'data', 'music_city_tech.csv'),
            'sqlite_table': 'mct_talks'
        })

    train_test_split_task = PythonOperator(
        task_id='train_test_split',
        python_callable=train_test_split)

    model_prep_task = BashOperator(
        task_id='model_prep',
        bash_command='mkdir -p "$AIRFLOW_HOME/models"')

    select_best_model_task = BranchPythonOperator(
        task_id='select_best_model',
        python_callable=select_best_model)

    # Workflow Definition
    [load_research_papers_task, load_mct_talks_task] \
        >> train_test_split_task \
        >> model_prep_task

    for classifier in CLASSIFIERS:
        model_prep_task \
            >> PythonOperator(
                task_id=f'train_{classifier.__name__}',
                python_callable=train_model,
                op_kwargs={'classifier': classifier}) \
            >> PythonOperator(
                task_id=f'test_{classifier.__name__}',
                python_callable=test_model,
                op_kwargs={'classifier': classifier}) \
            >> select_best_model_task \
            >> PythonOperator(
                task_id=f'predict_{classifier.__name__}',
                python_callable=predict,
                op_kwargs={'classifier': classifier})
