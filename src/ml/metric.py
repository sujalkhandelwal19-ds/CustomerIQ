from sklearn.metrics import f1_score, precision_score, recall_score
from src.entity.artifact_entity import ClassificationMetricArtifact


def calculate_metric(model, x_test, y_test) -> ClassificationMetricArtifact:
    predictions = model.predict(x_test)
    return ClassificationMetricArtifact(
        f1_score=float(f1_score(y_test, predictions, average="weighted")),
        precision_score=float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        recall_score=float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
    )
