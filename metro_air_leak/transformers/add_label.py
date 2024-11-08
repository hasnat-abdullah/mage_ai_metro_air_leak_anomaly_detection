from pandas import DataFrame
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs) -> DataFrame:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    timestamp_column = '_timestamp'
    
    data[timestamp_column] = data[timestamp_column].apply(pd.to_datetime, format = "%Y-%m-%d %H:%M:%S")
    
    failure_start_time = pd.to_datetime([
        "2020-04-18 00:00:00", 
        "2020-05-29 23:30:00", 
        "2020-06-05 10:00:00", 
        "2020-07-15 14:30:00"
    ])
    failure_end_time = pd.to_datetime([
        "2020-04-18 23:59:00", 
        "2020-05-30 06:00:00", 
        "2020-06-07 14:30:00", 
        "2020-07-15 19:00:00"
    ])
    
    # Create IntervalIndex for failure intervals including start and end point
    intervals = pd.IntervalIndex.from_tuples(list(zip(failure_start_time, failure_end_time)), closed='both')
    
    # if in the failure time interval then 1 else 0
    data['status'] = data[timestamp_column].apply(lambda x: 1 if any(x in interval for interval in intervals) else 0)
    
    print(data.head())
    status_counts = data['status'].value_counts()
    print("Total Positive (failure) sample with status = 1:", status_counts.get(1, 0))
    print("Total Negative (normal) sample with status = 0:", status_counts.get(0, 0))

    return data


@test
def test_output(output, *args) -> None:
    """
    testing datetime type.
    """
    assert pd.api.types.is_datetime64_any_dtype(output['_timestamp']), "Timestamp column conversion to datetime failed."
    assert 'status' in output.columns, "The 'status' column does not exist in the DataFrame."

    valid_status_values = {0, 1}
    invalid_values = output['status'].isin(valid_status_values) == False
    assert not invalid_values.any(), f"Found invalid values in 'status' column: {df['status'][invalid_values].tolist()}"