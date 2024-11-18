from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from pyod.models.xgbod import XGBOD  # PyOD's XGBOD model

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@custom
def transform_xgbod(df: DataFrame, *args, **kwargs):
    """
    Applies XGBOD for anomaly detection using XGBoost.

    Args:
        df: Input DataFrame
        args: The output from any upstream parent blocks (if applicable)

    Returns:
        Transformed DataFrame with predicted anomalies.
    """
    df = df.iloc[:, 2:]
    normal_data = df[df['_status'] == 0].drop(columns='_status')
    anomalous_data = df[df['_status'] == 1].drop(columns='_status')
    
    test_data = df.drop(columns='_status')

    # Create labels: 0 for normal, 1 for anomalous
    normal_labels = [0] * len(normal_data)
    anomalous_labels = [1] * len(anomalous_data)
    x_train = pd.concat([normal_data, anomalous_data])
    y_train = normal_labels + anomalous_labels

    xgbod = XGBOD(contamination=0.01, random_state=42)

    xgbod.fit(x_train, y_train)

    predictions = xgbod.predict(test_data)
    predicted_status = predictions 

    df['predicted_status'] = predicted_status

    report = classification_report(df['status'], df['predicted_status'], target_names=['Normal', 'Anomaly'])
    print(f"XGBOD Report:\n{report}")

    cm = confusion_matrix(df['status'], df['predicted_status'])
    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly']).plot()
    plt.title("Confusion Matrix for XGBOD")
    plt.show()

    # joblib.dump(xgbod, "xgbod_model.pkl")
    # print("XGBOD model saved as 'xgbod_model.pkl'")

    return df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'