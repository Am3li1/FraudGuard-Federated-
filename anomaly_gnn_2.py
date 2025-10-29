# anomaly_detection.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import json
from datetime import datetime
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

class GraphConstructor:
    """Constructs graphs from tabular data for GNN processing"""
    
    def __init__(self, k_neighbors=5, distance_threshold=0.5):
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.scaler = StandardScaler()
        
    def create_knn_graph(self, features):
        """Create k-nearest neighbor graph from feature matrix"""
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='ball_tree')
        nbrs.fit(features_normalized)
        distances, indices = nbrs.kneighbors(features_normalized)
        
        # Create edge list (excluding self-connections)
        edge_list = []
        edge_weights = []
        
        for i, neighbors in enumerate(indices):
            for j, neighbor_idx in enumerate(neighbors[1:]):  # Skip self (first neighbor)
                if distances[i][j+1] <= self.distance_threshold:
                    edge_list.append([i, neighbor_idx])
                    edge_weights.append(1.0 / (distances[i][j+1] + 1e-8))
        
        if len(edge_list) == 0:
            # Fallback: create minimal edges if no connections found
            n_nodes = len(features)
            for i in range(min(n_nodes-1, 10)):
                edge_list.append([i, i+1])
                edge_weights.append(1.0)
        
        return np.array(edge_list).T, np.array(edge_weights)
    
    def csv_to_graph(self, csv_file, target_column=None):
        """Convert CSV data to PyTorch Geometric graph format"""
        df = pd.read_csv(csv_file)
        print(f"Loaded CSV with shape: {df.shape}")
        
        # Handle categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            if col != target_column:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Separate features and target
        if target_column and target_column in df.columns:
            y = df[target_column].values
            X = df.drop(columns=[target_column]).values
        else:
            X = df.select_dtypes(include=[np.number]).values
            y = np.zeros(len(df))  # Dummy labels for unsupervised case
        
        print(f"Feature matrix shape: {X.shape}")
        
        # Create graph structure
        edge_index, edge_weights = self.create_knn_graph(X)
        
        # Convert to tensors
        x = torch.FloatTensor(self.scaler.transform(X))
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_weights)
        y = torch.LongTensor(y)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y), label_encoders

class GNNAnomalyDetector(nn.Module):
    """Graph Neural Network for anomaly detection"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_layers=3, dropout=0.3):
        super(GNNAnomalyDetector, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Attention layer for better anomaly detection
        self.attention = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Reconstruction head for unsupervised anomaly detection
        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # Graph convolution layers with residual connections
        h = x
        for i, conv in enumerate(self.conv_layers):
            h_new = F.relu(conv(h, edge_index))
            h_new = F.dropout(h_new, self.dropout, training=self.training)
            
            # Add residual connection if dimensions match
            if i > 0 and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
        
        # Apply attention mechanism
        h = self.attention(h, edge_index)
        
        # Global pooling for graph-level prediction
        if batch is not None:
            h_global = global_mean_pool(h, batch)
        else:
            h_global = torch.mean(h, dim=0, keepdim=True)
        
        # Classification output
        classification_out = self.classifier(h_global)
        
        # Reconstruction output
        reconstruction_out = self.reconstructor(h)
        
        return classification_out, reconstruction_out, h

class AnomalyReportGenerator:
    """Generates detailed reports of detected anomalies"""
    
    def __init__(self):
        self.anomaly_patterns = {}
        self.report_data = []
    
    def detect_anomalies_unsupervised(self, model, data_loader, threshold_percentile=95):
        """Detect anomalies using reconstruction error"""
        model.eval()
        anomalies = []
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch in data_loader:
                _, reconstruction, embeddings = model(batch)
                
                # Calculate reconstruction error
                recon_error = F.mse_loss(reconstruction, batch.x, reduction='none').mean(dim=1)
                reconstruction_errors.extend(recon_error.cpu().numpy())
        
        # Set threshold based on percentile
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        print(f"Anomaly threshold (reconstruction error): {threshold:.4f}")
        
        # Identify anomalies
        for i, error in enumerate(reconstruction_errors):
            if error > threshold:
                anomalies.append({
                    'index': i,
                    'reconstruction_error': float(error),
                    'anomaly_score': float(error / threshold),
                    'is_anomaly': True
                })
            else:
                anomalies.append({
                    'index': i,
                    'reconstruction_error': float(error),
                    'anomaly_score': float(error / threshold),
                    'is_anomaly': False
                })
        
        return anomalies, threshold
    
    def analyze_anomaly_patterns(self, anomalies, features, dataset_name):
        """Analyze patterns in detected anomalies"""
        anomaly_indices = [a['index'] for a in anomalies if a['is_anomaly']]
        
        if len(anomaly_indices) == 0:
            return {"pattern_type": "no_anomalies", "characteristics": {}}
        
        anomaly_features = features[anomaly_indices]
        normal_indices = [i for i in range(len(features)) if i not in anomaly_indices]
        normal_features = features[normal_indices] if normal_indices else np.array([])
        
        # Statistical analysis
        pattern_analysis = {
            "dataset": dataset_name,
            "anomaly_count": len(anomaly_indices),
            "anomaly_percentage": len(anomaly_indices) / len(features) * 100,
            "feature_statistics": {},
            "clustering_info": {}
        }
        
        # Feature-wise analysis
        for i in range(anomaly_features.shape[1]):
            anomaly_col = anomaly_features[:, i]
            normal_col = normal_features[:, i] if len(normal_features) > 0 else np.array([])
            
            pattern_analysis["feature_statistics"][f"feature_{i}"] = {
                "anomaly_mean": float(np.mean(anomaly_col)),
                "anomaly_std": float(np.std(anomaly_col)),
                "normal_mean": float(np.mean(normal_col)) if len(normal_col) > 0 else 0,
                "normal_std": float(np.std(normal_col)) if len(normal_col) > 0 else 0,
                "difference_ratio": float(abs(np.mean(anomaly_col) - np.mean(normal_col)) / (np.std(normal_col) + 1e-8)) if len(normal_col) > 0 else 0
            }
        
        # Clustering analysis
        if len(anomaly_features) > 1:
            clustering = DBSCAN(eps=0.5, min_samples=2)
            clusters = clustering.fit_predict(anomaly_features)
            pattern_analysis["clustering_info"] = {
                "n_clusters": len(set(clusters)) - (1 if -1 in clusters else 0),
                "noise_points": list(clusters).count(-1),
                "cluster_distribution": {str(c): list(clusters).count(c) for c in set(clusters)}
            }
        
        self.anomaly_patterns[dataset_name] = pattern_analysis
        return pattern_analysis
    
    def generate_report(self, dataset1_anomalies, dataset2_anomalies, dataset1_features, dataset2_features, output_file="anomaly_report.json"):
        """Generate comprehensive anomaly report"""
        
        # Analyze patterns for both datasets
        pattern1 = self.analyze_anomaly_patterns(dataset1_anomalies, dataset1_features, "dataset1")
        pattern2 = self.analyze_anomaly_patterns(dataset2_anomalies, dataset2_features, "dataset2")
        
        # Create comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_anomalies_dataset1": sum(1 for a in dataset1_anomalies if a['is_anomaly']),
                "total_anomalies_dataset2": sum(1 for a in dataset2_anomalies if a['is_anomaly']),
                "dataset1_anomaly_rate": pattern1["anomaly_percentage"],
                "dataset2_anomaly_rate": pattern2["anomaly_percentage"]
            },
            "dataset1_analysis": pattern1,
            "dataset2_analysis": pattern2,
            "anomaly_details": {
                "dataset1": dataset1_anomalies,
                "dataset2": dataset2_anomalies
            },
            "pattern_comparison": self._compare_patterns(pattern1, pattern2)
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Anomaly report saved to {output_file}")
        return report
    
    def _compare_patterns(self, pattern1, pattern2):
        """Compare anomaly patterns between datasets"""
        comparison = {
            "anomaly_rate_difference": abs(pattern1["anomaly_percentage"] - pattern2["anomaly_percentage"]),
            "feature_differences": {},
            "clustering_comparison": {
                "dataset1_clusters": pattern1["clustering_info"].get("n_clusters", 0),
                "dataset2_clusters": pattern2["clustering_info"].get("n_clusters", 0)
            }
        }
        
        # Compare feature statistics
        for feature in pattern1["feature_statistics"]:
            if feature in pattern2["feature_statistics"]:
                diff_ratio_1 = pattern1["feature_statistics"][feature]["difference_ratio"]
                diff_ratio_2 = pattern2["feature_statistics"][feature]["difference_ratio"]
                comparison["feature_differences"][feature] = {
                    "pattern_similarity": 1 / (1 + abs(diff_ratio_1 - diff_ratio_2))
                }
        
        return comparison

def main_detection_workflow(dataset1_path, dataset2_path, output_dir="output", k_neighbors=5, threshold_percentile=95):
    """Complete workflow for GNN anomaly detection"""
    
    print("=== GNN Anomaly Detection Workflow ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create graph constructor
    print("Step 1: Creating graph structures...")
    graph_constructor = GraphConstructor(k_neighbors=k_neighbors)
    
    # Convert CSV files to graph format
    try:
        graph1, encoders1 = graph_constructor.csv_to_graph(dataset1_path)
        graph_constructor_2 = GraphConstructor(k_neighbors=k_neighbors)  # New instance for dataset 2
        graph2, encoders2 = graph_constructor_2.csv_to_graph(dataset2_path)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None
    
    print(f"Dataset 1: {graph1.x.shape[0]} nodes, {graph1.edge_index.shape[1]} edges")
    print(f"Dataset 2: {graph2.x.shape[0]} nodes, {graph2.edge_index.shape[1]} edges")
    
    # Step 2: Initialize anomaly detection models
    print("\nStep 2: Initializing anomaly detection models...")
    input_dim = graph1.x.shape[1]
    
    model1 = GNNAnomalyDetector(input_dim)
    model2 = GNNAnomalyDetector(input_dim)
    
    # Create data loaders
    loader1 = DataLoader([graph1], batch_size=1)
    loader2 = DataLoader([graph2], batch_size=1)
    
    # Step 3: Detect anomalies (unsupervised)
    print("\nStep 3: Detecting anomalies...")
    report_generator = AnomalyReportGenerator()
    
    anomalies1, threshold1 = report_generator.detect_anomalies_unsupervised(
        model1, loader1, threshold_percentile=threshold_percentile
    )
    anomalies2, threshold2 = report_generator.detect_anomalies_unsupervised(
        model2, loader2, threshold_percentile=threshold_percentile
    )
    
    # Step 4: Generate comprehensive report
    print("\nStep 4: Generating anomaly report...")
    report_file = os.path.join(output_dir, "anomaly_report.json")
    report = report_generator.generate_report(
        anomalies1, anomalies2, 
        graph1.x.numpy(), graph2.x.numpy(),
        report_file
    )
    
    print(f"Found {report['summary']['total_anomalies_dataset1']} anomalies in dataset 1 ({report['summary']['dataset1_anomaly_rate']:.2f}%)")
    print(f"Found {report['summary']['total_anomalies_dataset2']} anomalies in dataset 2 ({report['summary']['dataset2_anomaly_rate']:.2f}%)")
    
    # Step 5: Create labeled training dataset
    print("\nStep 5: Creating labeled training dataset...")
    
    # Load original datasets
    df1 = pd.read_csv(dataset1_path)
    df2 = pd.read_csv(dataset2_path)
    
    # Add anomaly labels based on detection results
    labels1 = [1 if a['is_anomaly'] else 0 for a in anomalies1]
    labels2 = [2 if a['is_anomaly'] else 0 for a in anomalies2]  # Different pattern
    
    df1['anomaly_type'] = labels1
    df2['anomaly_type'] = labels2
    df2['dataset_source'] = 'dataset2'  # Add source identifier
    df1['dataset_source'] = 'dataset1'
    
    # Combine datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    training_data_file = os.path.join(output_dir, "training_data_with_labels.csv")
    combined_df.to_csv(training_data_file, index=False)
    
    print(f"Training data saved to {training_data_file}")
    
    # Save metadata for training
    metadata = {
        "input_dim": input_dim,
        "dataset1_path": dataset1_path,
        "dataset2_path": dataset2_path,
        "threshold1": float(threshold1),
        "threshold2": float(threshold2),
        "k_neighbors": k_neighbors,
        "threshold_percentile": threshold_percentile,
        "timestamp": datetime.now().isoformat()
    }
    
    metadata_file = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training metadata saved to {metadata_file}")
    
    print("\n=== Detection Phase Complete ===")
    print("Files generated:")
    print(f"1. {report_file} - Detailed anomaly analysis")
    print(f"2. {training_data_file} - Labeled training data")
    print(f"3. {metadata_file} - Training metadata")
    print("\nNext step: Run 'python train_gnn_model.py' to train the main GNN model")
    
    return report, training_data_file

def main():
    parser = argparse.ArgumentParser(description='GNN Anomaly Detection - Detection Phase')
    parser.add_argument('dataset1', help='Path to first CSV dataset')
    parser.add_argument('dataset2', help='Path to second CSV dataset')
    parser.add_argument('--output-dir', default='output', help='Output directory for results')
    parser.add_argument('--k-neighbors', type=int, default=5, help='Number of neighbors for graph construction')
    parser.add_argument('--threshold-percentile', type=float, default=95, help='Percentile threshold for anomaly detection')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.dataset1):
        print(f"Error: Dataset 1 file '{args.dataset1}' not found!")
        return
    
    if not os.path.exists(args.dataset2):
        print(f"Error: Dataset 2 file '{args.dataset2}' not found!")
        return
    
    print(f"Dataset 1: {args.dataset1}")
    print(f"Dataset 2: {args.dataset2}")
    print(f"Output directory: {args.output_dir}")
    print(f"K-neighbors: {args.k_neighbors}")
    print(f"Threshold percentile: {args.threshold_percentile}")
    
    # Run detection workflow
    try:
        report, training_file = main_detection_workflow(
            args.dataset1, 
            args.dataset2, 
            args.output_dir,
            args.k_neighbors,
            args.threshold_percentile
        )
        
        if report is not None:
            print("\nDetection completed successfully!")
        else:
            print("\nDetection failed!")
            
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()