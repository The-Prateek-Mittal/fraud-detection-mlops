# 🚨 Fraud Detection MLOps

An **end-to-end Machine Learning Operations (MLOps) project** for fraud detection, implementing the complete machine learning lifecycle — from data processing and model training to **containerization, automated CI/CD, Kubernetes deployment, and interactive model inference using Streamlit**.

The project demonstrates how a machine learning model can be transformed from a development experiment into a reproducible and deployable ML system.

---

## 📌 Project Overview

Traditional machine learning projects often stop after training a model.

This project goes further by implementing the operational infrastructure required to build, test, package, deploy, and maintain a machine learning application.

The complete workflow is:

```text
                ┌──────────────────┐
                │   Data / Dataset │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Data Processing  │
                │ & Preprocessing  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Model Training   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Model Evaluation │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Docker Container │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ GitHub Actions   │
                │     CI/CD        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    Kubernetes    │
                │    Deployment    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    Streamlit     │
                │   Web Interface  │
                └──────────────────┘
```

---

# 🎯 Objectives

The primary objectives of this project are to:

* Build a machine learning system for fraud detection
* Create a reproducible ML development environment
* Separate ML source code from deployment infrastructure
* Containerize the application using Docker
* Automate the development workflow using GitHub Actions
* Deploy the application using Kubernetes
* Provide an interactive Streamlit interface for predictions
* Organize the project according to MLOps principles
* Demonstrate the complete ML lifecycle from development to deployment

---

# 🧠 Machine Learning Pipeline

The machine learning workflow follows:

```text
Raw Data
   │
   ▼
Data Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Train / Test Split
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Model Serialization
   │
   ▼
Deployment
```

The trained model is stored as an artifact and subsequently used by the application for inference.

---

# 🏗️ System Architecture

The project is divided into multiple layers.

```text
┌─────────────────────────────────────────────────────────────┐
│                        USER / CLIENT                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       STREAMLIT APP                         │
│                  Interactive Prediction UI                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     ML INFERENCE LAYER                      │
│                 Load Model → Predict Fraud                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     TRAINED MODEL                           │
│                 Serialized Model Artifact                   │
└─────────────────────────────────────────────────────────────┘


                    MLOps Infrastructure
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
     GitHub Actions         Docker            Kubernetes
          │                   │                   │
          ▼                   ▼                   ▼
       CI/CD             Containerization      Deployment
```

---

# 🔄 End-to-End MLOps Workflow

The complete lifecycle can be represented as:

```text
┌─────────────┐
│ Development │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Git      │
│   Commit    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ GitHub Actions  │
│   CI Pipeline   │
└────────┬────────┘
         │
         ├──────────────► Tests
         │
         ├──────────────► Dependency Installation
         │
         └──────────────► Build / Validation
                            │
                            ▼
                     ┌──────────────┐
                     │    Docker    │
                     │     Image    │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ Kubernetes   │
                     │  Deployment  │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Streamlit   │
                     │  Application │
                     └──────┬───────┘
                            │
                            ▼
                     Fraud Prediction
```

---

# 🛠️ Technology Stack

| Technology         | Purpose                                      |
| ------------------ | -------------------------------------------- |
| **Python**         | Machine learning and application development |
| **Scikit-learn**   | Machine learning                             |
| **Pandas**         | Data processing                              |
| **NumPy**          | Numerical computation                        |
| **Streamlit**      | Interactive ML web application               |
| **Docker**         | Containerization                             |
| **Kubernetes**     | Container orchestration                      |
| **GitHub Actions** | CI/CD automation                             |
| **Git**            | Version control                              |
| **Pytest**         | Testing                                      |

---

# 🐳 Docker

Docker is used to package the ML application and its dependencies into a reproducible container.

Without containerization:

```text
Developer Machine
      │
      ├── Python Version
      ├── Libraries
      ├── OS Dependencies
      └── Configuration
```

This can lead to:

> "It works on my machine."

Docker solves this by packaging the application environment:

```text
┌────────────────────────────┐
│       Docker Container     │
│                            │
│  Application               │
│  Python                    │
│  Dependencies              │
│  Model                     │
│  Configuration             │
│                            │
└────────────────────────────┘
```

### Build the Docker Image

```bash
docker build -t fraud-detection-mlops .
```

### Run the Container

```bash
docker run -p 8501:8501 fraud-detection-mlops
```

The Streamlit application can then be accessed through:

```text
http://localhost:8501
```

---

# ☸️ Kubernetes Deployment

Kubernetes is used to orchestrate the containerized ML application.

The repository contains a dedicated:

```text
k8s/
```

directory for Kubernetes deployment configuration.

The deployment architecture can be represented as:

```text
                 Kubernetes Cluster
                        │
                        ▼
              ┌──────────────────┐
              │    Deployment    │
              └────────┬─────────┘
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
        ┌─────────┐         ┌─────────┐
        │  Pod 1  │         │  Pod 2  │
        │   ML    │         │   ML    │
        │  App    │         │  App    │
        └─────────┘         └─────────┘
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
                 Kubernetes
                   Service
                       │
                       ▼
                   Streamlit
```

This allows the application to be managed as a containerized workload instead of being tied to a single machine.

---

# 🔁 CI/CD with GitHub Actions

GitHub Actions automates the software delivery workflow.

The repository contains:

```text
.github/
└── workflows/
```

for CI/CD configuration.

The intended workflow is:

```text
Developer pushes code
        │
        ▼
GitHub Repository
        │
        ▼
GitHub Actions
        │
        ├── Install dependencies
        │
        ├── Run tests
        │
        ├── Validate application
        │
        └── Build deployment artifacts
                │
                ▼
             Docker
                │
                ▼
          Kubernetes
                │
                ▼
          ML Application
```

This reduces the amount of manual work required between development and deployment.

---

# 🧪 Testing

The project includes a dedicated:

```text
tests/
```

directory for testing the ML/application components.

Testing is an important part of the CI pipeline because changes to the model or application should be validated before deployment.

Tests can be executed locally using:

```bash
pytest
```

---

# 🖥️ Streamlit Application

Streamlit provides an interactive interface for interacting with the trained fraud detection model.

The application allows a user to provide transaction-related inputs and receive a prediction from the trained model.

Conceptually:

```text
User Input
    │
    ▼
Streamlit Interface
    │
    ▼
Input Preprocessing
    │
    ▼
Trained ML Model
    │
    ▼
Prediction
    │
    ▼
Fraud / Legitimate Result
```

This provides a simple interface for demonstrating the deployed machine learning system without requiring users to interact directly with Python code.

---

# 📁 Project Structure

The repository is organized into separate components for application code, data, models, source code, tests, Kubernetes infrastructure, and CI/CD.

```text
fraud-detection-mlops/
│
├── .github/
│   └── workflows/
│       └── CI/CD configuration
│
├── app/
│   └── Streamlit application
│
├── data/
│   └── Dataset / data resources
│
├── k8s/
│   └── Kubernetes deployment configuration
│
├── models/
│   └── Trained model artifacts
│
├── src/
│   └── Machine learning source code
│
├── tests/
│   └── Automated tests
│
├── Dockerfile
│
├── requirements.txt
│
├── requirements-local.txt
│
└── README.md
```

---

# 🚀 Getting Started

## Prerequisites

Install the following software:

* Python 3.x
* Git
* Docker
* Kubernetes
* kubectl

For local development, a Python virtual environment is recommended.

---

## 1. Clone the Repository

```bash
git clone https://github.com/The-Prateek-Mittal/fraud-detection-mlops.git
```

Move into the project:

```bash
cd fraud-detection-mlops
```

---

# 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For local development:

```bash
pip install -r requirements-local.txt
```

---

# 4. Run the Application Locally

Run the Streamlit application using:

```bash
streamlit run app/<streamlit_app_file>.py
```

The terminal will provide the local URL for accessing the application.

---

# 5. Run Tests

```bash
pytest
```

---

# 🐳 Run with Docker

Build the image:

```bash
docker build -t fraud-detection-mlops .
```

Run it:

```bash
docker run -p 8501:8501 fraud-detection-mlops
```

Open:

```text
http://localhost:8501
```

---

# ☸️ Deploy with Kubernetes

Make sure Kubernetes is running and `kubectl` is configured.

Check the cluster:

```bash
kubectl get nodes
```

Apply the Kubernetes configurations:

```bash
kubectl apply -f k8s/
```

Check deployed resources:

```bash
kubectl get pods
```

Check services:

```bash
kubectl get services
```

Check deployments:

```bash
kubectl get deployments
```

---

# 🔍 Monitoring the Deployment

Useful Kubernetes commands:

### View Pods

```bash
kubectl get pods
```

### Detailed Pod Information

```bash
kubectl describe pod <pod-name>
```

### View Logs

```bash
kubectl logs <pod-name>
```

### View Services

```bash
kubectl get svc
```

### View Deployments

```bash
kubectl get deployments
```

These commands allow the deployed ML application to be inspected and debugged.

---

# 🔐 MLOps Principles Demonstrated

This project demonstrates several important MLOps principles.

### 1. Reproducibility

Dependencies and deployment environments are defined explicitly.

```text
Code
 +
Dependencies
 +
Docker
 =
Reproducible Environment
```

### 2. Automation

GitHub Actions automates repetitive CI/CD tasks.

```text
Code Change
    ↓
Automated Pipeline
    ↓
Testing
    ↓
Build
    ↓
Deployment
```

### 3. Containerization

Docker isolates the ML application from the host environment.

### 4. Orchestration

Kubernetes manages the deployed containerized application.

### 5. Testing

Automated tests help validate changes before deployment.

### 6. Separation of Concerns

The project separates:

```text
Data
ML Code
Application
Models
Tests
Infrastructure
CI/CD
```

This makes the system easier to maintain and extend.

---

# 🔄 ML Lifecycle

The complete lifecycle implemented in this project can be summarized as:

```text
       ┌─────────────┐
       │     Data    │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │ Preprocess  │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │   Training  │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │ Evaluation  │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │ Model Saved │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │    Docker   │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │ GitHub      │
       │ Actions     │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │ Kubernetes  │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │  Streamlit  │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │ Prediction  │
       └─────────────┘
```

---

# 💡 Why This Project Matters

A machine learning model by itself is only one component of a production ML system.

This project focuses on the engineering required to operationalize that model.

Instead of:

```text
Dataset → Notebook → Model
```

the project implements:

```text
Data
 ↓
ML Pipeline
 ↓
Testing
 ↓
Containerization
 ↓
CI/CD
 ↓
Kubernetes
 ↓
Application
 ↓
Prediction
```

This makes the project a practical demonstration of **Machine Learning Engineering and MLOps**.

---

# 🚧 Future Improvements

Potential extensions include:

* MLflow experiment tracking
* DVC-based dataset versioning
* Automated model retraining
* Model performance gates in CI/CD
* Model registry
* Prometheus monitoring
* Grafana dashboards
* Data drift detection
* Model drift detection
* Automated rollback
* Cloud deployment
* Kubernetes Horizontal Pod Autoscaling
* Docker image registry integration
* Security scanning of Docker images
* API-based model serving
* Automated model promotion

---

# 📚 Key Learning Outcomes

Through this project, the following concepts were implemented and explored:

### Machine Learning

* Data preprocessing
* Feature engineering
* Model training
* Model evaluation
* Model persistence
* Inference

### Software Engineering

* Modular project structure
* Testing
* Dependency management
* Version control

### DevOps

* Docker
* Containerization
* CI/CD
* GitHub Actions

### Cloud-Native / Infrastructure

* Kubernetes
* Pods
* Deployments
* Services
* Container orchestration

### ML Application Development

* Streamlit
* Interactive model inference
* ML model integration

---

# 👨‍💻 Author

**Prateek Mittal**

B.Tech — Computing and Data Science

GitHub:
https://github.com/The-Prateek-Mittal

LinkedIn:
https://www.linkedin.com/in/prateek-mittal-75278b206

---

# ⭐ Project Summary

> **An end-to-end fraud detection MLOps system implementing the complete machine learning lifecycle with automated CI/CD, Docker containerization, Kubernetes deployment, automated testing, and a Streamlit-based inference interface.**

```text
┌─────────────────────────────────────────────────────┐
│                 FRAUD DETECTION ML                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Data → Training → Testing → Docker → CI/CD        │
│                              │                      │
│                              ▼                      │
│                         Kubernetes                 │
│                              │                      │
│                              ▼                      │
│                         Streamlit                  │
│                              │                      │
│                              ▼                      │
│                        Prediction                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

⭐ If you find this project useful, consider starring the repository.
