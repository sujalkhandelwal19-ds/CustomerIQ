# CustomerIQ — Customer Segmentation

CustomerIQ is an end-to-end customer segmentation system. It ingests customer data from MongoDB, validates the dataset, engineers customer-level features, preprocesses the data, creates customer segments with PCA + K-Means, trains a Logistic Regression classifier to reproduce those segments for new customers, evaluates the model, stores the trained model in AWS S3, and exposes prediction through FastAPI.

## Architecture

1. **MongoDB** — source customer data.
2. **Data ingestion** — exports the collection and creates a train/test split.
3. **Data validation** — checks the exact schema and detects excessive train/test distribution drift.
4. **Feature engineering** — derives age, children, family-related and spending/promotion features.
5. **Preprocessing** — imputes missing values, scales standard numeric features, and applies a power transform to skewed spending features.
6. **Clustering** — PCA and K-Means are fitted **only on the training set**. The fitted K-Means model is then used to assign test-set cluster labels.
7. **Classification** — Logistic Regression learns to predict the cluster from the original engineered features.
8. **Evaluation** — weighted F1, precision and recall are calculated on the held-out test set.
9. **Model registry/storage** — the serialized model can be pushed to AWS S3.
10. **Prediction API** — FastAPI loads the model from S3 and predicts the customer cluster.

## Why two ML stages?

K-Means is used to discover customer groups without a pre-existing target label. Once the clusters are created, their cluster IDs become the supervised target for Logistic Regression. This allows the application to predict a segment for a new customer without rerunning K-Means during every API request.

A critical implementation detail is that PCA and K-Means are fitted once on training data and reused for the test set. This prevents the train and test sets from receiving independently learned cluster definitions.

## Features used for prediction

- Age
- Education
- Marital Status
- Parental Status
- Children
- Income
- Total Spending
- Days as Customer
- Recency
- Wines
- Fruits
- Meat
- Fish
- Sweets
- Gold
- Web Purchases
- Catalog Purchases
- Store Purchases
- Discount Purchases
- Total Promotions
- Website Visits per Month

## Tech stack

- Python 3.11
- FastAPI
- Pandas / NumPy
- Scikit-learn
- MongoDB
- AWS S3
- Docker
- GitHub Actions

## Local setup

### 1. Clone

```bash
git clone https://github.com/sujalkhandelwal19-ds/CustomerIQ.git
cd CustomerIQ
```

### 2. Create an environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

MongoDB:

```text
MONGO_DB_URL=<your-mongodb-connection-string>
```

AWS:

```text
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
AWS_DEFAULT_REGION=ap-south-1
MODEL_BUCKET_NAME=<your-s3-model-bucket>
```

Do not commit credentials to Git.

### 5. Start the API

```bash
python app.py
```

The API runs on:

```text
http://localhost:5000
```

Health check:

```text
GET /health
```

Training:

```text
POST /train
```

Prediction UI:

```text
GET /
POST /
```

## Docker

Build:

```bash
docker build -t customeriq .
```

Run:

```bash
docker run --rm -p 5000:5000 \
  -e MONGO_DB_URL="<your-mongodb-url>" \
  -e AWS_ACCESS_KEY_ID="<your-access-key>" \
  -e AWS_SECRET_ACCESS_KEY="<your-secret-key>" \
  -e AWS_DEFAULT_REGION="ap-south-1" \
  -e MODEL_BUCKET_NAME="<your-bucket>" \
  customeriq
```

## Model evaluation

The classifier is tuned using GridSearchCV on the training set. The held-out test set is used only for final evaluation.

Reported metrics:

- Weighted F1 score
- Weighted precision
- Weighted recall

The training pipeline fails when the final weighted F1 score is below the configured minimum threshold.

## Project structure

```text
CustomerIQ/
├── app.py
├── Dockerfile
├── requirements.txt
├── config/
├── src/
│   ├── components/
│   ├── configuration/
│   ├── constant/
│   ├── data_access/
│   ├── entity/
│   ├── ml/
│   ├── pipeline/
│   └── utils/
├── templates/
├── static/
└── .github/
```

## Important implementation notes

- The preprocessing object is fitted on training data and reused during prediction.
- PCA and K-Means are fitted only on training data.
- Test labels are generated using the training-fitted K-Means model.
- The classifier is evaluated on untouched test data.
- Model metrics are calculated from actual predictions; they are not hard-coded.
- The project no longer depends on the obsolete `neuro_mf` or Evidently development dependency.

## License

This project is for educational and portfolio purposes.
