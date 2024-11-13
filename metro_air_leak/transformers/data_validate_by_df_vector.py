import pandas as pd
import numpy as np
from pandas import DataFrame
from datetime import datetime


def vectorized_validation(df):
    valid_TP2 = df['TP2'].between(-100, 100)
    valid_TP3 = df['TP3'] >= 0
    valid_DV_pressure = df['DV_pressure'].between(-10, 10)
    valid_Oil_temperature = df['Oil_temperature'] >= 0
    valid_Motor_current = df['Motor_current'] >= 0
    valid_COMP = df['COMP'] >= 0
    valid_DV_eletric = df['DV_eletric'] >= 0
    valid_Towers = df['Towers'] >= 0
    valid_MPG = df['MPG'] >= 0
    valid_LPS = df['LPS'] >= 0
    valid_Pressure_switch = df['Pressure_switch'] >= 0
    valid_Oil_level = df['Oil_level'] >= 0
    valid_Caudal_impulses = df['Caudal_impulses'] >= 0

    valid = (
        valid_TP2 &
        valid_TP3 &
        valid_DV_pressure &
        valid_Oil_temperature &
        valid_Motor_current &
        valid_COMP &
        valid_DV_eletric &
        valid_Towers &
        valid_MPG &
        valid_LPS &
        valid_Pressure_switch &
        valid_Oil_level &
        valid_Caudal_impulses
    )
    return df[valid] # retrun only valid row

@transformer
def transform(df:DataFrame, *args, **kwargs):
    validated_df = vectorized_validation(df)
    print(validated_df.head())
    return validated_df


@test
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'