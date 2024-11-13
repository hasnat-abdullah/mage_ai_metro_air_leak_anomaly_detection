import pandas as pd
import numpy as np
from pandas import DataFrame
from datetime import datetime


from pydantic import BaseModel, confloat, conint, ValidationError
from datetime import datetime
from sqlalchemy import create_engine

# Define the Pydantic model
class CompressorData(BaseModel):
    timestamp: datetime
    TP2: confloat(ge=-100, le=100)
    TP3: confloat(ge=0)
    H1: confloat()
    DV_pressure: confloat(ge=-10, le=10)
    Reservoirs: confloat()
    Oil_temperature: confloat(ge=0)
    Motor_current: confloat(ge=0)
    COMP: conint(ge=0)
    DV_eletric: conint(ge=0)
    Towers: conint(ge=0)
    MPG: conint(ge=0)
    LPS: conint(ge=0)
    Pressure_switch: conint(ge=0)
    Oil_level: conint(ge=0)
    Caudal_impulses: conint(ge=0)

    class Config:
        json_encoders = {
            datetime: lambda v: v.strftime('%Y-%m-%d %H:%M:%S')
        }
def validate_and_process_data(df):
    validated_data = []
    skipped_faulty_row_count = 0
    
    for index, row in df.iterrows():
        
        try:
            validated_record = CompressorData(**row.to_dict()).dict()
            validated_data.append(validated_record)
        except ValidationError as e:
            skipped_faulty_row_count += 1
            print(f"Validation error in row {index}: {e}")
    
    return validated_data

@transformer
def transform(df:DataFrame, *args, **kwargs):
    validated_data = validate_and_process_data(df)
    validated_df = pd.DataFrame(validated_data)
    print(validated_df.head())
    return validated_df


@test
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'