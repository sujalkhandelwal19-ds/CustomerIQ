import os

PRED_SCHEMA_FILE_PATH = os.path.join("config", "prediction_schema.yaml")
PREDICTION_DATA_BUCKET = os.getenv("PREDICTION_DATA_BUCKET", "customer-iq-data")
PREDICTION_INPUT_FILE_NAME = "prediction_input.csv"
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME", "customer-iq-models")
PREDICTION_OUTPUT_FILE_NAME = "prediction_output.csv"
