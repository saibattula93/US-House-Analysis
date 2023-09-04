import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['State',
                                'Land_Availability',
                                'Economic_Conditions',
                                'Consumer_Confidence',
                                'Demographic_Trends',
                                'Location',
                                'Amenities']
            numerical_cols = ['Housing_Inventory',
                            'Construction_Costs',
                            'Interest_Rates',
                            'Population_Growth',
                            'House_Area_(sqft)']
                                        
            # Define the custom ranking for each ordinal variable
            State_categories = ['Pennsylvania', 'Kentucky', 'South Dakota', 'Texas', 'Tennessee',
                                'Illinois', 'Oregon', 'Wisconsin', 'Washington', 'Utah',
                                'Connecticut', 'Oklahoma', 'Maryland', 'Hawaii', 'West Virginia',
                                'Indiana', 'Maine', 'Rhode Island', 'Florida', 'Georgia',
                                'Alabama', 'Arkansas', 'Mississippi', 'New York', 'Iowa',
                                'Michigan', 'North Dakota', 'Alaska', 'Colorado', 'Virginia',
                                'Kansas', 'Ohio', 'Nebraska', 'South Carolina', 'New Hampshire',
                                'Wyoming', 'Louisiana', 'California', 'Nevada', 'Idaho',
                                'Missouri', 'Delaware', 'Massachusetts', 'New Jersey',
                                'New Mexico', 'North Carolina', 'Minnesota', 'Vermont', 'Montana',
                                'Arizona']
            Land_Availability_categories = ['Abundant', 'Limited']
            Economic_Conditions_categories = ['Moderate', 'Weak', 'Strong']
            Consumer_Confidence_categories = ['High', 'Moderate', 'Low']
            Demographic_Trends_categories = ['Aging Population', 'Millennial Buyers']
            Location_categories = ['Urban', 'Suburban']
            Amenities_categories = ['Good Schools', 'Transport']

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())                
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder',OrdinalEncoder(categories=[State_categories,
                                                              Land_Availability_categories,
                                                              Economic_Conditions_categories,
                                                              Consumer_Confidence_categories,
                                                              Demographic_Trends_categories,
                                                              Location_categories,
                                                              Amenities_categories])),
                ('scaler',StandardScaler())
                ]
            )

            logging.info(f'Categorical Columns : {categorical_cols}')
            logging.info(f'Numerical Columns   : {numerical_cols}')

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.info('Exception occured in Data Transformation Phase')
            raise CustomException(e,sys)
        
    def initate_data_transformation(self,train_path,test_path):

        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'House_Price_($)'
            drop_columns = [target_column_name,'Date']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            logging.info('Exception occured in initiate_data_transformation function')
            raise CustomException(e,sys)