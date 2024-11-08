from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def get_selected_features_x_df(x_train, y_train, rf:RandomForestClassifier):
    num_features_to_select = 7

    num_features = len(x_train.columns.tolist())
    looped_x_train = x_train.copy()

    while num_features> num_features_to_select:
        n1 = num_features-1
        n2 = num_features-2
        sfe = SequentialFeatureSelector(estimator=rf, n_features_to_select=n1, direction = 'forward', n_jobs = -1)
        sbe = SequentialFeatureSelector(estimator=rf, n_features_to_select=n2, direction = 'backward', n_jobs= -1)
        #Eliminating Features with Forward pass
        sfe.fit(looped_x_train, y_train)
        sfe_features = looped_x_train.columns[sfe.support_].tolist()
        looped_x_train = looped_x_train[sfe_features]
        #Eliminating Features with Backward pass
        sbe.fit(looped_x_train, y_train)
        sbe_features = looped_x_train.columns[sbe.support_].tolist()
        looped_x_train = looped_x_train[sbe_features]

        num_features = len(looped_x_train.columns.tolist())

    selected_features = looped_x_train.columns
    print(f"List of selected features:\n {', '.join(selected_features)}")
    return looped_x_train

@custom
def transform_feature_selection(df: DataFrame, *args, **kwargs):
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
    
    # x_train = get_selected_features_x_df(x_train, y_train, rf)
    
    
    #train randorm forest
    rf.fit(x_train, y_train)
    selected_features = x_train.columns
    #EValuate the result of new features
    y_pred_forest = rf.predict(x_test[selected_features])
    accuracy = accuracy_score(y_test, y_pred_forest)

    print("Random Forest Classifier:")
    print(classification_report(y_test, y_pred_forest))
    conf_matrix_forest = confusion_matrix(y_test, y_pred_forest)
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
