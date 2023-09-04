import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass




class CustomData:
    def __init__(self,
        State: str, 
        Housing_Inventory: int, 
        Construction_Costs : int,
        Land_Availability : str, 
        Interest_Rates : int,
        Economic_Conditions : str,
        Population_Growth : int, 
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
        





        