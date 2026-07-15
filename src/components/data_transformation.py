import sys
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact, DataIngestionArtifact

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, read_yaml_file, save_numpy_array_data


class DataTransformation:
    def __init__(
            self, 
            data_ingestion_artifact: DataIngestionArtifact,
            data_transformation_config: DataTransformationConfig,
            data_validation_artifact: DataValidationArtifact
        ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise MyException(e, sys)
        

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            return MyException(e, sys)
        
    def custom_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply custom preprocessing steps.
        """
        logging.info("Started custom preprocessing")
        df = df.copy()
        # Drop ID column
        logging.info("Dropping Unnecessary Columns")
        drop_col = self._schema_config["drop_columns"]
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
            logging.info(f"Dropped column: {drop_col}")
        # Map Gender
        logging.info("Mapping 'Gender' column to binary values.")
        df["Gender"] = df["Gender"].map({
            "Female": 0,
            "Male": 1
        }).astype(int)
        # One-Hot Encoding
        logging.info("Applying One-Hot Encoding.")
        df = pd.get_dummies(df, drop_first=True)
        # Rename columns
        logging.info("Renaming encoded columns.")
        df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        }, inplace=True)
        # Convert dummy columns to int
        dummy_cols = [
            "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_gt_2_Years",
            "Vehicle_Damage_Yes"
        ]
        logging.info("Converting dummy columns to integer type.")
        for col in dummy_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        logging.info("Custom preprocessing completed successfully.")
        return df
    
    def get_data_transformer_object(self):
        """
        Creates and returns the preprocessing pipeline.
        """
        try:
            logging.info("Entered get_data_transformer_object method.")
            num_features = self._schema_config["num_features"]
            mm_features = self._schema_config["mm_features"]
            logging.info("Loaded numerical and MinMax feature lists from schema.")
            preprocessing_pipeline = Pipeline([
                (
                    "custom_preprocessing",
                    FunctionTransformer(
                        self.custom_preprocessing,
                        validate=False
                    )
                ),
                (
                    "scaling",
                    ColumnTransformer(
                        transformers=[
                            ("standard", StandardScaler(), num_features),
                            ("minmax", MinMaxScaler(), mm_features)
                        ],
                        remainder="passthrough"
                    )
                )
            ])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return preprocessing_pipeline
        
        except Exception as e:
            logging.exception("Exception occured while creating preprocessing pipeline")
            raise MyException(e, sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline
        """
        logging.info("Data Transformation Started!!")
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            # Load train and test data
            logging.info("Loading Train-Test data")
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data looaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor Object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")

            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTEENN applied to train-test df.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            logging.info("Saving transformation object and transformed files.")
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            logging.info("Transformation object and transformed files saved")
            logging.info("Data transformation completed successfully")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            raise MyException(e, sys)