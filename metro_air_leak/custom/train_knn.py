from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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
    x = df.iloc[:, 2:-1]
    y = df.iloc[:, -1]
    print(x.head)
    print(y.head)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=5)
    #train KNN
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)

    print(f"KNN report:\n{classification_report(y_test, y_pred)} ")
    labels = ['Normal', 'Anomaly']
    cm =confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
