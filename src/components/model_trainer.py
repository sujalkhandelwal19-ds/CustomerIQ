import sys
import os
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, load_numpy_array_data
from src.ml.model.estimator import CustomerSegmentationModel


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.utils = MainUtils()

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method")
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)

            # Hyperparameter tuning is performed only on the training set.
            model = LogisticRegression(
                max_iter=1000,
                multi_class="auto",
                solver="lbfgs",
                random_state=42,
            )
            param_grid = {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            }
            search = GridSearchCV(
                model,
                param_grid=param_grid,
                scoring="f1_weighted",
                cv=3,
                n_jobs=-1,
            )
            search.fit(x_train, y_train)
            best_model = search.best_estimator_

            predictions = best_model.predict(x_test)
            metrics = ClassificationMetricArtifact(
                f1_score=float(f1_score(y_test, predictions, average="weighted")),
                precision_score=float(
                    precision_score(y_test, predictions, average="weighted", zero_division=0)
                ),
                recall_score=float(
                    recall_score(y_test, predictions, average="weighted", zero_division=0)
                ),
            )

            if metrics.f1_score < self.model_trainer_config.expected_accuracy:
                raise ValueError(
                    f"Best model F1 score {metrics.f1_score:.4f} is below "
                    f"required threshold {self.model_trainer_config.expected_accuracy:.4f}"
                )

            preprocessing_obj = self.utils.load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )
            trained_model = CustomerSegmentationModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model,
            )

            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=trained_model,
            )

            artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metrics,
            )
            logging.info(
                "Model training completed. best_params=%s, metrics=%s",
                search.best_params_,
                metrics,
            )
            return artifact

        except Exception as e:
            raise CustomerException(e, sys) from e
