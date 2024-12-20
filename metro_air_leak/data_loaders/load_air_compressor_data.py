from mage_ai.settings.repo import get_repo_path
from os import path
import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_file(*args, **kwargs):
    """
    Optimized template for loading data from filesystem.
    Load data from a single CSV file using pandas for better performance.
    Drops any unnamed columns after loading.

    Docs: https://docs.mage.ai/design/data-loading#fileio
    """
    filepath = path.join(get_repo_path(), 'raw_data/MetroPT3AirCompressor_data.csv')

    df = pd.read_csv(filepath)

    # Drop unnamed columns (those with names starting with 'unnamed')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(df.columns)

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert isinstance(output, pd.DataFrame), 'The output is not a DataFrame'