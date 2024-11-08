
# Metro Air Leak Anomaly Detection

This project focuses on detecting anomalies in air compressor data to support predictive maintenance. By identifying abnormal patterns in the data, the model aims to detect potential failures in advance, helping to reduce downtime and improve system reliability.

## Dataset
The dataset used in this project is the [Metropt 3 Dataset](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset) from the UCI Machine Learning Repository. It contains sensor readings from air compressors over several months, providing a range of features such as pressure, motor current, and oil temperature, which help in identifying anomalies and predicting compressor failures.

## Models and Approach
This project leverages various machine learning algorithms, implemented using [Mage.ai](https://www.mage.ai), to detect potential failures based on sensor data. The following algorithms are applied:
- **Isolation Forest**: A tree-based model effective for high-dimensional anomaly detection.
- **Autoencoder**: A neural network model used to detect unusual patterns by reconstructing normal behavior.
- **LSTM (Long Short-Term Memory)**: A recurrent neural network (RNN) model suited for time-series anomaly detection.

These models are trained to recognize normal operation patterns and identify deviations that might signal a potential failure.


## Getting Started
1. Clone this repository:
   ```bash
   git clone https://github.com/hasnat-abdullah/mage_ai_metro_air_leak_anomaly_detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd metro-air-leak-anomaly-detection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the project using Mage.ai:
   ```bash
   mage start
   ```


## Contributing
Contributions to improve the project are welcome. Please feel free to fork the repository and make pull requests.

## License
This project is licensed under the MIT License.

## Acknowledgments
- The dataset is sourced from the UCI Machine Learning Repository.
- Developed with Mage.ai for efficient data pipeline and model development.
