import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        State: str, 
        Housing_Inventory: int, 
        Construction_Costs : int,
        Land_Availability : str, 
        Interest_Rates : float,
        Economic_Conditions : str,
        Population_Growth : float, 
        Consumer_Confidence : str, 
        Demographic_Trends : str,
        House_Area_sqft : int, 
        Location , 
        Amenities: str):


        self.State = State

        self.Housing_Inventory = Housing_Inventory

        self.Construction_Costs = Construction_Costs

        self.Land_Availability = Land_Availability

        self.Interest_Rates = Interest_Rates

        self.Economic_Conditions = Economic_Conditions

        self.Population_Growth = Population_Growth

        self.Consumer_Confidence = Consumer_Confidence
        self.Demographic_Trends = Demographic_Trends
        self.House_Area_sqft = House_Area_sqft

        self.Location = Location

        self.Amenities = Amenities

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "State": [self.State],
                "Housing_Inventory": [self.Housing_Inventory],
                "Construction_Costs": [self.Construction_Costs],
                "Land_Availability": [self.Land_Availability],
                "Interest_Rates": [self.Interest_Rates],
                "Economic_Conditions": [self.Economic_Conditions],
                "Population_Growth": [self.Population_Growth],
                "Consumer_Confidence": [self.Consumer_Confidence],
                "Demographic_Trends": [self.Demographic_Trends],
                "House_Area_sqft": [self.House_Area_sqft],
                "Location": [self.Location],
                "Amenities": [self.Amenities]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)



        