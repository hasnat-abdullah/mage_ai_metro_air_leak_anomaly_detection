from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
    x = df.iloc[:, 2:-1]
    y = df.iloc[:, -1]
    print(x.head)
    print(y.head)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    rf_params = {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
    rf = RandomForestClassifier(oob_score=True, random_state=42, **rf_params)
    
    #train randorm forest
    rf.fit(x_train, y_train)
    y_pred_forest = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_forest)

    print(f"Random forest report: \n{classification_report(y_test, y_pred_forest)}")
    cm = confusion_matrix(y_test, y_pred_forest)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    disp_forest = ConfusionMatrixDisplay(conf_matrix_forest, display_labels=rf.classes_)
    disp_forest.plot(cmap='Blues', values_format='d')
    plt.show()
    


    return {}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
