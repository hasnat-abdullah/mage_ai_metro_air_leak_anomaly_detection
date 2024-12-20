from pandas import DataFrame
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,roc_auc_score,roc_curve, confusion_matrix, ConfusionMatrixDisplay, precision_score,recall_score,f1_score
if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_knn(df: DataFrame, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    x = df.iloc[:, 1:-1]
    print(x)
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    print('start training')
    knn = KNeighborsClassifier(n_neighbors=3)
    print("Start training")
    knn.fit(x_train_scaled, y_train)

    # Predict on the test set
    print("Start predicting")
    y_pred = knn.predict(x_test_scaled)

    # print(f"KNN report:\n{classification_report(y_test, y_pred)} ")
    # labels = ['Normal', 'Anomaly']
    # cm =confusion_matrix(y_test, y_pred)
    # cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    # print(cm_df)
    # Accuracy
    try:
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        # Precision
        precision = precision_score(y_test, y_pred)
        print(f'Precision: {precision:.2f}')

        # Recall
        recall = recall_score(y_test, y_pred)
        print(f'Recall: {recall:.2f}')

        # F1-Score
        f1 = f1_score(y_test, y_pred)
        print(f'F1-Score: {f1:.2f}')

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print('Confusion Matrix:')
        print(cm)

        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f'ROC AUC: {roc_auc:.2f}')
        # Save the trained knn model
        joblib.dump(knn, "knn_model_v1.pkl")
        print("knn model saves as 'knn_model_v1.pkl' ")
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()
    except Exception as ex:
        print(ex)

    # Save the trained knn model
    joblib.dump(knn, "knn_model_v1.pkl")
    print("knn model saves as 'knn_model_v1.pkl' ")


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
