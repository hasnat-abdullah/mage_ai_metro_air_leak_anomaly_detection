from typing import Dict, List
from pandas import DataFrame
import joblib
import sys
import logging

logger = logging.getLogger(__name__)

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

def load_model():
    """Load the pre-trained KNN model."""
 
    model = joblib.load('knn_model_v1.pkl')
    logger.info("Model loaded successfully.")
    return model

def predict_anomaly(data: Dict, model) -> int:
    """Predict anomaly using the trained model."""
    df = DataFrame([data])
    timestamp = df.loc[0, '_timestamp']

    df = df.drop(columns=['_timestamp', 'tp2'], errors='ignore')

    prediction = model.predict(df)
    logger.info(f"Prediction for timestamp {timestamp}: {prediction[0]}")
    return timestamp, prediction[0]

@transformer
def transform(messages: List[Dict], *args, **kwargs) -> DataFrame:
    """
    Processes Kafka messages, predicts anomalies, and returns a DataFrame.

    Args:
        messages: List of messages in the stream.

    Returns:
        DataFrame with '_timestamp' and 'prediction' columns.
    """
    model = load_model()
    results = []

    logger.info("Kafka Consumer Started...")
    
    for message in messages:
        try:
            data = message.get('data', {}) 
            logger.debug(f"Received message: {data}")

            timestamp, prediction = predict_anomaly(data, model)
            results.append({'_timestamp': timestamp, 'prediction': prediction})
        except Exception as e:
            sys.stdout.write(str(e))
            logger.error(f"Error processing message: {e}")

    logger.info("Kafka Consumer Finished Processing Messages.")
    predicted_df = DataFrame(results, columns=['_timestamp', 'prediction'])
    sys.stdout.write(predicted_df.to_string())
    return predicted_df