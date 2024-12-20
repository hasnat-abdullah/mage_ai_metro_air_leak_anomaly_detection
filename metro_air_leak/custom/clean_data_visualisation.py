import seaborn as sb
from pandas import DataFrame

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(df: DataFrame,*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # data corelation
    # df.corr().round(2)
    # sb.heatmap(df.corr().round(2),annot=False )
    # sb.pairplot(df,  y_vars = ['status'] , plot_kws=  {'alpha' : 0.1})
    # # visualize Outlier to check if our outlier removal not worked
    # sb.set(rc={'figure.figsize':(20,8.20)})
    # sb.boxplot(df)

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
