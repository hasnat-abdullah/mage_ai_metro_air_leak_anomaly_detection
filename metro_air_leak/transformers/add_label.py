from pandas import DataFrame
import pandas as pd
import numpy as np

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    comvert timestamp column from str to datetime type.
    Add another column called 'status' where 0 = Normal and 1 = Failure.
    Dataset provided the failure start and end time. based on this data we will figure out the status value.

    Args:
        df: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        DataFrame
    """
    timestamp_column = 'timestamp'
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], format="%Y-%m-%d %H:%M:%S")
    
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
    
    # Create an array of boolean series for each interval
    conditions = [(df[timestamp_column] >= start) & (df[timestamp_column] <= end) 
                  for start, end in zip(failure_start_time, failure_end_time)]
    
    # Combine all conditions with logical OR; 1=Failure, 0=Normal
    df['status'] = np.where(np.logical_or.reduce(conditions), 1, 0)
    
    status_counts = df['status'].value_counts()
    print("Total Positive (failure) sample with status = 1:", status_counts.get(1, 0))
    print("Total Negative (normal) sample with status = 0:", status_counts.get(0, 0))
    print(df.columns)
    return df


@test
def test_output(output, *args) -> None:
    """
    testing datetime type.
    """
    assert pd.api.types.is_datetime64_any_dtype(output['timestamp']), "Timestamp column conversion to datetime failed."
    assert 'status' in output.columns, "The 'status' column does not exist in the DataFrame."

    valid_status_values = {0, 1}
    invalid_values = output['status'].isin(valid_status_values) == False
    assert not invalid_values.any(), f"Found invalid values in 'status' column: {df['status'][invalid_values].tolist()}"