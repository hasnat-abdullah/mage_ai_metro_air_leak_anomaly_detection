from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@custom
def transform_random_forest(df: DataFrame, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Split the data into normal and anomalous data
    df = df.iloc[:, 2:]
    normal_data = df[df['status'] == 0].drop(columns='status')  # Drop status column for training
    anomalous_data = df[df['status'] == 1].drop(columns='status')

    # Combine normal and anomalous data for testing
    test_data = df.drop(columns='status')



    isolation_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

    # Train isolation forest
    isolation_forest.fit(normal_data)
    predictions = isolation_forest.predict(test_data)
    predicted_status = (predictions == -1).astype(int)
    # Add predicted status to the DataFrame for comparison
    df['predicted_status'] = predicted_status

    # Classification report
    report = classification_report(df['status'], df['predicted_status'], target_names=['Normal', 'Anomaly'])
    print(f"isolation forest:\n{report}")

    # Confusion Matrix
    labels = ['Normal', 'Anomaly']
    cm = confusion_matrix(df['status'], df['predicted_status'])
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
