from pandas import DataFrame
import seaborn as sb
import matplotlib.pyplot as plt


if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Data overview to know deeper about data

    Returns:
        None
    """
    print(f'data overview:\n{df.describe().round(2)}')
    print(f'Columns:\n{df.columns}')
    # visualize Outlier
    # plt.figure(figsize=(12, 8))
    # sb.boxplot(df)
    # plt.xticks(rotation=90)  # Rotate column names if needed
    # plt.show()
    return df



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'