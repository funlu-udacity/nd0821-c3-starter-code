"""
Census Class

Author: Ferruh Unlu

Date: 2/5/2022

"""


from pydantic import BaseModel

class Census(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: float
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 42,
                "workclass": 'Private',
                "fnlgt": 52789,
                "education": "Masters",
                "education_num": 17,
                "marital_status": "Married",
                "occupation": "Data-Scientist",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 4174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }
