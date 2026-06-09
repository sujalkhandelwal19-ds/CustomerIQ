# 🛍️ CustomerIQ — Customer Categorization System

> **A complete end-to-end Machine Learning pipeline to predict and categorize customers based on their demographic and transactional data.**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green?style=for-the-badge&logo=fastapi)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-brightgreen?style=for-the-badge&logo=mongodb)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)
![AWS](https://img.shields.io/badge/AWS-S3-orange?style=for-the-badge&logo=amazonaws)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-black?style=for-the-badge&logo=githubactions)

</div>

---

## 📌 Problem Statement

In today's competitive market, understanding your customer is everything. This project builds a machine learning system that predicts the **personality/category of a customer** using their personal and purchase details.

This is highly useful for:
- 🏬 Malls & retail stores
- 🛒 E-commerce platforms
- 🏢 Product-based companies

Based on a customer's **personal details** and **purchase behavior**, we:
1. **Cluster** them into meaningful segments using unsupervised learning
2. **Predict** the cluster of a new customer using classification techniques

---

## 💡 Solution Proposed

The approach leverages a full ML pipeline:
- Cluster existing customers using **K-Means**
- Train a **Random Forest / Logistic Regression** classifier on the labeled clusters
- Deploy the model as an API for real-time prediction

**GitHub Repository:** [PWskills-DataScienceTeam/Customer-Categorizer](https://github.com/PWskills-DataScienceTeam/Customer-Categorizer)

**Dataset:** [Marketing Campaign Dataset](https://github.com/entbappy/Branching-tutorial/blob/master/marketing_campaign.zip)

---

## 🔄 End-to-End Data Flow

```mermaid
flowchart TD
    A(["Raw Data - Marketing Campaign CSV"]) --> B["Data Ingestion - Load from MongoDB and S3"]
    B --> C["Data Validation - Schema and Quality Checks"]
    C --> D["Data Transformation - Feature Engineering and Scaling"]
    D --> E["K-Means Clustering - Unsupervised Segmentation"]
    E --> F["Cluster Labels Assigned to Each Customer"]
    F --> G["Model Training - Random Forest or Logistic Regression"]
    G --> H["Model Evaluation - Accuracy, F1, GridSearchCV"]
    H --> I{"Model Accepted?"}
    I -- Yes --> J["Model Pusher - Save to AWS S3"]
    I -- No --> G
    J --> K(["FastAPI Endpoint - Real-Time Predictions"])

    style A fill:#4A90D9,color:#fff,stroke:#2c5f8a
    style K fill:#27AE60,color:#fff,stroke:#1a7a42
    style I fill:#F39C12,color:#fff,stroke:#b07a0f
    style E fill:#8E44AD,color:#fff,stroke:#6c3483
    style G fill:#E74C3C,color:#fff,stroke:#b03a2e
```

---

## 🧠 ML Pipeline Architecture

```mermaid
flowchart LR
    subgraph INGESTION["📥 Data Ingestion"]
        A1[MongoDB Atlas] --> A2[Raw DataFrame]
        A3[AWS S3 Bucket] --> A2
    end

    subgraph PREPROCESSING["🔧 Preprocessing"]
        B1[Handle Missing Values]
        B2[Encode Categorical Cols]
        B3[Feature Scaling]
        B1 --> B2 --> B3
    end

    subgraph CLUSTERING["🔵 Clustering"]
        C1["Elbow Method - Optimal K"]
        C2[K-Means Fit]
        C3[Assign Cluster Labels]
        C1 --> C2 --> C3
    end

    subgraph CLASSIFICATION["🤖 Classification"]
        D1["Train and Test Split"]
        D2["Random Forest or Logistic Regression"]
        D3["GridSearchCV - Hyperparameter Tuning"]
        D4[Best Model Selected]
        D1 --> D2 --> D3 --> D4
    end

    subgraph SERVING["🚀 Serving"]
        E1[FastAPI App]
        E2["/predict endpoint"]
        E3[Customer Cluster Returned]
        E1 --> E2 --> E3
    end

    INGESTION --> PREPROCESSING --> CLUSTERING --> CLASSIFICATION --> SERVING

    style INGESTION fill:#1a3a5c,color:#fff
    style PREPROCESSING fill:#1a4a2e,color:#fff
    style CLUSTERING fill:#3b1a5c,color:#fff
    style CLASSIFICATION fill:#5c1a1a,color:#fff
    style SERVING fill:#1a4a4a,color:#fff
```

---

## ☁️ Deployment Architecture

```mermaid
flowchart TD
    DEV(["Developer - Local Machine"]) -->|git push| GH["GitHub Repository"]

    GH -->|Trigger| CI["GitHub Actions - CI/CD Pipeline"]

    CI --> TEST["Run Tests and Linting"]
    TEST -->|Pass| BUILD["Docker Build - Create Image"]
    BUILD --> PUSH["Push Image to Docker Registry"]

    PUSH --> DEPLOY{"Cloud Deploy"}

    DEPLOY --> AWS["AWS EC2 - Production Server"]
    DEPLOY --> AZURE["Azure - Cloud Instance"]

    AWS --> APP(["FastAPI App - Running on Port 5000"])
    AZURE --> APP

    APP -->|Reads Model| S3["AWS S3 - Model Storage"]
    APP -->|Fetches Data| MONGO["MongoDB Atlas - Customer Database"]
    APP -->|Monitoring| EV["Evidently - Drift Detection"]

    USER(["End User or Client App"]) -->|POST predict| APP
    APP -->|Customer Cluster| USER

    style DEV fill:#2c3e50,color:#fff
    style USER fill:#2c3e50,color:#fff
    style CI fill:#e67e22,color:#fff
    style APP fill:#27ae60,color:#fff
    style S3 fill:#e74c3c,color:#fff
    style MONGO fill:#27ae60,color:#fff
    style EV fill:#8e44ad,color:#fff
```

---

## 🔃 Training vs Prediction Pipeline

```mermaid
flowchart LR
    subgraph TRAIN["Training Pipeline - route: /train"]
        T1[Ingest Data] --> T2[Validate] --> T3[Transform]
        T3 --> T4[Cluster with K-Means]
        T4 --> T5[Train Classifier]
        T5 --> T6[Evaluate Model]
        T6 --> T7[Push to S3]
    end

    subgraph PREDICT["Prediction Pipeline - route: /predict"]
        P1[Receive Customer Input] --> P2[Load Model from S3]
        P2 --> P3[Preprocess Input]
        P3 --> P4[Predict Cluster]
        P4 --> P5[Return Category to User]
    end

    style TRAIN fill:#1a3a5c,color:#fff
    style PREDICT fill:#1a4a2e,color:#fff
```

---

## 🧰 Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.7+ |
| ML Framework | Scikit-learn, XGBoost |
| API Framework | FastAPI + Uvicorn |
| Database | MongoDB (Atlas) |
| Cloud Storage | AWS S3 |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Platform | Azure / AWS |

---

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| `scikit-learn` | ML algorithms (K-Means, Logistic Regression) |
| `xgboost` | High-performance gradient boosting |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server for FastAPI |
| `pymongo` | MongoDB driver for Python |
| `boto3` | AWS SDK – interact with S3, EC2 |
| `evidently` | ML model monitoring & data drift detection |
| `imbalanced-learn` | Handle imbalanced datasets (SMOTE) |
| `dill` | Enhanced Python object serialization |
| `python-dotenv` | Load environment variables from `.env` |
| `jinja2` | Templating engine for FastAPI |
| `python-multipart` | File upload support in APIs |
| `neuro_mf` | Neural network meta-feature extractor |
| `from-root` | Dynamic project root directory finder |
| `watchfiles` | Real-time file monitoring / auto-reload |

---

## 🗂️ Project Architecture

### `src/` — Main Package

```
src/
├── components/
│   ├── data_ingestion.py        # Load raw data from MongoDB / S3
│   ├── data_validation.py       # Schema & data quality checks
│   ├── data_transformation.py   # Feature engineering & preprocessing
│   ├── data_clustering.py       # K-Means clustering
│   ├── model_trainer.py         # Train classification model
│   ├── model_evaluation.py      # Evaluate model performance
│   └── model_pusher.py          # Push model to cloud storage
│
├── logger.py                    # Custom logging
├── exception.py                 # Custom exception handling
└── pipeline/
    ├── training_pipeline.py     # End-to-end training flow
    └── prediction_pipeline.py   # Real-time prediction flow
```

---

## 🤖 Models Used

| Stage | Model | Purpose |
|---|---|---|
| Clustering | **K-Means** | Segment customers into clusters |
| Classification | **Logistic Regression** / **Random Forest** | Predict cluster of new customers |
| Optimization | **GridSearchCV** | Hyperparameter tuning |

---

## ⚙️ How to Run

### Prerequisites
- MongoDB Atlas account with the dataset loaded
- AWS account (for S3 storage)
- Python 3.7+

### Step 1 — Clone the Repository

```bash
git clone https://github.com/sujalkhandelwal19-ds/CustomerIQ.git
cd CustomerIQ
```

### Step 2 — Create and Activate Virtual Environment

```bash
conda create --prefix venv python=3.7 -y
conda activate venv/
```

### Step 3 — Install Requirements

```bash
pip install -r requirements.txt
```

### Step 4 — Set Environment Variables

```bash
export AWS_ACCESS_KEY_ID=<your_aws_access_key>
export AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>
export AWS_DEFAULT_REGION=<your_aws_region>
export MONGODB_URL=<your_mongodb_connection_string>
```

> 💡 On Windows, use `set` instead of `export`, or use a `.env` file with `python-dotenv`.

### Step 5 — Run the Application

```bash
python app.py
```

### Step 6 — Train the Model

```
GET http://localhost:5000/train
```

### Step 7 — Make Predictions

```
GET http://localhost:5000/predict
```

---

## 🐳 Docker Deployment

### Build the Docker Image

```bash
docker build \
  --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> \
  --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> \
  --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION> \
  --build-arg MONGODB_URL=<MONGODB_URL> \
  .
```

### Run the Container

```bash
docker run -d -p 5000:5000 <IMAGE_NAME>
```

---

## ☁️ Cloud & MLOps Infrastructure

| Component | Service |
|---|---|
| Model Storage | AWS S3 |
| Database | MongoDB Atlas |
| Deployment | Azure / AWS EC2 |
| CI/CD Pipeline | GitHub Actions |
| Experiment Tracking | Evidently (data drift) |

---

## ✅ Conclusion

The **CustomerIQ** system is a production-ready ML solution that:
- Segments customers intelligently using unsupervised learning
- Classifies new customers in real-time via a REST API
- Is fully containerized and deployable on cloud platforms
- Follows MLOps best practices with CI/CD, logging, and monitoring

---

<div align="center">
  <b>Built with ❤️ by the team | PW Skills</b>
</div>
