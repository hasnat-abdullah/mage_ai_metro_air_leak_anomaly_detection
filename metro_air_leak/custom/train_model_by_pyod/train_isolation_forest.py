from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from pyod.models.iforest import IForest  # PyOD Isolation Forest

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@custom
def transform_random_forest_pyod(df: DataFrame, *args, **kwargs):
    """
    Applies PyOD's Isolation Forest model for anomaly detection.

    Args:
        df: Input DataFrame
        args: The output from any upstream parent blocks (if applicable)

    Returns:
        Transformed DataFrame with predicted anomalies.
    """
    df = df.iloc[:, 2:]
    normal_data = df[df['status'] == 0].drop(columns='status')
    test_data = df.drop(columns='status')

    isolation_forest = IForest(contamination=0.01, random_state=42)

    isolation_forest.fit(normal_data)
    predictions = isolation_forest.predict(test_data) 
    predicted_status = predictions 

    df['predicted_status'] = predicted_status

    report = classification_report(df['status'], df['predicted_status'], target_names=['Normal', 'Anomaly'])
    print(f"Isolation Forest Report:\n{report}")

    cm = confusion_matrix(df['status'], df['predicted_status'])
    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly']).plot()
    plt.title("Confusion Matrix for PyOD Isolation Forest")
    plt.show()

    
    # joblib.dump(isolation_forest, "isolation_forest_pyod_model.pkl")
    # print("Isolation Forest model saved as 'isolation_forest_pyod_model.pkl'")

    return df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'