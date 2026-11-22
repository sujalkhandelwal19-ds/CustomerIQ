import json
import sys
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from pandas import DataFrame

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig

from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, write_yaml_file


class DataValidation:
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig ):
        
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        
        self.utils = MainUtils()

        self._schema_config = self.utils.read_schema_config_file()

    def validate_schema_columns(self, dataframe: DataFrame) -> bool:
        try:
            expected = list(self._schema_config["columns"].keys())
            actual = list(dataframe.columns)
            status = actual == expected
            logging.info(f"Schema validation status: {status}")
            if not status:
                logging.error(f"Expected columns: {expected}; received: {actual}")
            return status
        except Exception as e:
            raise CustomerException(e, sys) from e

    def validate_dataset_schema_columns(self, train_set, test_set) -> Tuple[bool, bool]:
        """
        Method Name :   validate_dataset_schema_columns
        Description :   This method validates the schema for schema columns for both train and test set 
        
        Output      :   True or False value is returned based on the schema 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info(
            "Entered validate_dataset_schema_columns method of Data_Validation class"
        )

        try:
            logging.info("Validating dataset schema columns")

            train_schema_status = self.validate_schema_columns(train_set)

            logging.info("Validated dataset schema columns on the train set")

            test_schema_status = self.validate_schema_columns(test_set)

            logging.info("Validated dataset schema columns on the test set")

            logging.info("Validated dataset schema columns")

            return train_schema_status, test_schema_status

        except Exception as e:
            raise CustomerException(e, sys) from e

    

    def detect_dataset_drift(
        self, reference_df: DataFrame, current_df: DataFrame
    ) -> bool:
        """Detect meaningful train/test distribution drift without external reporting libraries."""
        try:
            drifted = 0
            total = len(reference_df.columns)
            for column in reference_df.columns:
                ref = reference_df[column].dropna()
                cur = current_df[column].dropna()
                if len(ref) == 0 or len(cur) == 0:
                    continue
                if pd.api.types.is_numeric_dtype(ref):
                    p_value = ks_2samp(ref, cur).pvalue
                else:
                    ref_dist = ref.value_counts(normalize=True)
                    cur_dist = cur.value_counts(normalize=True)
                    categories = set(ref_dist.index).union(cur_dist.index)
                    distance = sum(
                        abs(ref_dist.get(cat, 0.0) - cur_dist.get(cat, 0.0))
                        for cat in categories
                    ) / 2.0
                    p_value = 0.0 if distance > 0.30 else 1.0
                if p_value < 0.05:
                    drifted += 1

            drift_ratio = drifted / total if total else 0.0
            logging.info(f"Dataset drift: {drifted}/{total} features ({drift_ratio:.2%})")
            return drift_ratio > 0.30
        except Exception as e:
            raise CustomerException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_data_validation method of Data_Validation class")

        try:
            logging.info("Initiated data validation for the dataset")

            train_df, test_df = (DataValidation.read_data(file_path = self.data_ingestion_artifact.trained_file_path),
                                DataValidation.read_data(file_path = self.data_ingestion_artifact.test_file_path))
            
            
            
            drift = self.detect_dataset_drift(train_df, test_df)

            (
                schema_train_col_status,
                schema_test_col_status,
            ) = self.validate_dataset_schema_columns(train_set=train_df, test_set=test_df)

            logging.info(
                f"Schema train cols status is {schema_train_col_status} and schema test cols status is {schema_test_col_status}"
            )

            logging.info("Validated dataset schema columns")

            

            if (
                schema_train_col_status is True
                and schema_test_col_status is True
                and drift is False
            ):
                logging.info("Dataset schema validation completed")

                validation_status = True
            else:
                validation_status = False
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e
