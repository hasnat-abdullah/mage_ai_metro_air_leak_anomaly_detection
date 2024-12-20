from pandas import DataFrame
import joblib
from sklearn.metrics import classification_report, confusion_matrix


if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_run_model_knn(df:DataFrame, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    timestamps = df['_timestamp']
    y = df['_status']
    df = df.drop(['_timestamp', 'tp2', '_status'], axis=1)

    knn_model = joblib.load('metro_air_leak/models/knn_model_v1.pkl')

    y_pred = knn_model.predict(df)


    print(f"Classification Report:\n{classification_report(y, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

    result_df = DataFrame({
        'timestamp': timestamps,
        'prediction': y_pred
    })
    
    return result_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
