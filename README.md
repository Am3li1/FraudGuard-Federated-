# FraudGuard Federated
🧠 GNN-Based Anomaly Detection Pipeline

This project implements a Graph Neural Network (GNN) model to detect anomalies across multiple datasets, with a special focus on fraud detection in financial data.
It transforms raw tabular data into graph structures, identifies irregular patterns, and continuously improves through federated model updates.

⚙️ This project is currently in progress.

🚀 Features

📊 Automatic Graph Conversion – Converts tabular CSV data into graph structures using k-nearest neighbors (k-NN)

🧠 Unsupervised GNN Anomaly Detection – Detects anomalies based on structural and relational patterns

📑 Anomaly Report Generation – Saves detected anomalies into structured reports for each dataset

🌐 Multi-Dataset Support – Supports multiple CSV files as input for batch anomaly analysis

🔁 Simulated Real-Time Analysis – Streams simulated real-time data through the GNN model to detect fraudulent patterns dynamically

🤝 Collaborative Model Training (Federated Setup) – Aggregates anomaly reports from multiple financial institutions to train a global GNN model, ensuring:

Privacy-preserving collaboration

Model updates across all participants

Continuous improvement in fraud detection accuracy

🧩 How It Works

Data Loading

Each CSV dataset is read and preprocessed.

Missing values and outliers are handled automatically.

Graph Construction

Each data instance (row) is represented as a node.

Nodes are connected based on feature similarity using the k-nearest neighbors algorithm.

Anomaly Detection

The GNN learns embeddings that capture both feature and structural relationships.

Nodes with irregular embedding patterns are flagged as anomalies (potential fraud).

Report Generation

For each dataset, an anomaly report (anomaly_report_<filename>.csv) is generated.

These reports are stored for model training and performance review.

Global Model Training

Each participating institution trains a local GNN model using its data and anomaly reports.

These reports are aggregated to train a global GNN model, which is distributed back to all institutions for improved, unified fraud detection.

Real-Time Simulation

The system simulates real-time data flow through the GNN pipeline.

The model continuously identifies suspicious patterns and updates anomaly logs in real time.

🛠️ Requirements

Install all dependencies using:

pip install -r requirements.txt

