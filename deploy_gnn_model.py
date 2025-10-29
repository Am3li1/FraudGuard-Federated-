# deploy_gnn_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import json
import argparse
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the GNN models
from anomaly_gnn_2 import GraphConstructor
from gnn_training import NodeLevelGNNAnomalyDetector

class GNNDeploymentEngine:
    """Deploy trained GNN model to new CSV files for anomaly detection"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.scaler = None
        self.label_encoders = {}
        self.load_model()
    
    def load_model(self):
        """Load the trained model and metadata"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Extract model parameters
        input_dim = checkpoint.get('input_dim', 10)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        num_classes = checkpoint.get('num_classes', 3)
        model_class = checkpoint.get('model_class', 'NodeLevelGNNAnomalyDetector')
        
        # Initialize the correct model class
        if model_class == 'NodeLevelGNNAnomalyDetector':
            self.model = NodeLevelGNNAnomalyDetector(input_dim, hidden_dim, num_classes)
        else:
            # Fallback to NodeLevel for compatibility
            self.model = NodeLevelGNNAnomalyDetector(input_dim, hidden_dim, num_classes)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load metadata
        self.metadata = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_classes': num_classes,
            'model_class': model_class,
            'training_timestamp': checkpoint.get('timestamp', 'unknown'),
            'test_accuracy': checkpoint.get('test_accuracy', 'unknown')
        }
        
        # Load label encoders if available
        if 'label_encoders' in checkpoint:
            self.label_encoders = checkpoint['label_encoders']
        
        print(f"Model loaded successfully!")
        print(f"Model class: {model_class}")
        print(f"Input dimension: {input_dim}")
        print(f"Model test accuracy: {self.metadata['test_accuracy']}")
    
    def preprocess_csv(self, csv_path):
        """Preprocess CSV file for model input"""
        df = pd.read_csv(csv_path)
        original_df = df.copy()  # Keep original for output
        
        print(f"Loaded CSV with shape: {df.shape}")
        
        # Remove known label columns if they exist
        columns_to_remove = ['anomaly_type', 'dataset_source', 'is_anomaly']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"Removed column: {col}")
        
        # Handle categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col in self.label_encoders:
                # Use existing encoder
                try:
                    # Handle unseen categories
                    unique_values = df[col].unique()
                    known_classes = self.label_encoders[col]
                    
                    # Map known values and assign new label for unknown
                    df[col] = df[col].apply(lambda x: known_classes.index(x) if x in known_classes else len(known_classes))
                except Exception as e:
                    print(f"Warning: Could not apply saved encoder for {col}, using new encoder")
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
            else:
                # Create new encoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Ensure we have the right number of features
        expected_features = self.metadata['input_dim']
        if df.shape[1] != expected_features:
            print(f"Warning: Expected {expected_features} features, got {df.shape[1]}")
            if df.shape[1] > expected_features:
                # Take first N columns
                df = df.iloc[:, :expected_features]
                print(f"Truncated to {expected_features} features")
            else:
                # Pad with zeros
                for i in range(expected_features - df.shape[1]):
                    df[f'padding_{i}'] = 0
                print(f"Padded to {expected_features} features")
        
        return df.values, original_df
    
    def create_graph_from_features(self, features, k_neighbors=5):
        """Create graph structure from features"""
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Create k-NN graph
        n_neighbors = min(k_neighbors + 1, len(features))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        nbrs.fit(features_normalized)
        distances, indices = nbrs.kneighbors(features_normalized)
        
        # Create edge list
        edge_list = []
        edge_weights = []
        
        for i, neighbors in enumerate(indices):
            for j, neighbor_idx in enumerate(neighbors[1:]):  # Skip self
                edge_list.append([i, neighbor_idx])
                edge_weights.append(1.0 / (distances[i][j+1] + 1e-8))
        
        # Fallback if no edges created
        if len(edge_list) == 0:
            for i in range(min(len(features) - 1, 10)):
                edge_list.append([i, i+1])
                edge_weights.append(1.0)
        
        # Convert to tensors
        edge_index = torch.LongTensor(np.array(edge_list).T)
        edge_attr = torch.FloatTensor(edge_weights)
        x = torch.FloatTensor(features_normalized)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), scaler
    
    def predict_anomalies(self, csv_path, output_path=None, threshold_multiplier=1.0, k_neighbors=5):
        """Predict anomalies in a new CSV file"""
        
        print(f"\n=== Anomaly Detection on {csv_path} ===")
        
        # Preprocess data
        features, original_df = self.preprocess_csv(csv_path)
        
        # Create graph
        graph_data, scaler = self.create_graph_from_features(features, k_neighbors)
        
        # Make predictions
        with torch.no_grad():
            data_loader = DataLoader([graph_data], batch_size=1)
            
            predictions = []
            reconstruction_errors = []
            node_embeddings = []
            
            for batch in data_loader:
                classification_out, reconstruction_out, embeddings = self.model(batch)
                
                # Get class probabilities for each node
                probabilities = F.softmax(classification_out, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                
                # Calculate reconstruction errors for each node
                recon_errors = F.mse_loss(reconstruction_out, batch.x, reduction='none').mean(dim=1)
                
                # Store results for each node
                for i in range(len(batch.x)):
                    node_pred = {
                        'node_index': i,
                        'predicted_class': predicted_classes[i].item(),
                        'normal_probability': probabilities[i][0].item(),
                        'anomaly_pattern1_probability': probabilities[i][1].item(),
                        'anomaly_pattern2_probability': probabilities[i][2].item(),
                        'total_anomaly_probability': probabilities[i][1].item() + probabilities[i][2].item(),
                        'reconstruction_error': recon_errors[i].item(),
                        'is_anomaly': predicted_classes[i].item() > 0
                    }
                    predictions.append(node_pred)
                    reconstruction_errors.append(recon_errors[i].item())
                    node_embeddings.append(embeddings[i].cpu().numpy())
        
        # Apply additional threshold filtering based on reconstruction error
        recon_threshold = np.percentile(reconstruction_errors, 95) * threshold_multiplier
        
        # Update anomaly predictions based on both classification and reconstruction
        anomaly_count_class = sum(1 for p in predictions if p['is_anomaly'])
        
        for i, pred in enumerate(predictions):
            # Combine classification and reconstruction-based detection
            high_recon_error = pred['reconstruction_error'] > recon_threshold
            high_anomaly_prob = pred['total_anomaly_probability'] > 0.5
            
            pred['is_anomaly_combined'] = pred['is_anomaly'] or high_recon_error
            pred['anomaly_confidence'] = min(1.0, pred['total_anomaly_probability'] + 
                                           (pred['reconstruction_error'] / recon_threshold))
        
        anomaly_count_combined = sum(1 for p in predictions if p['is_anomaly_combined'])
        
        # Create results DataFrame
        results_df = pd.DataFrame(predictions)
        
        # Add original data
        for col in original_df.columns:
            if col not in results_df.columns:
                results_df[col] = original_df[col].values
        
        # Summary statistics
        summary = {
            'total_records': len(predictions),
            'anomalies_by_classification': anomaly_count_class,
            'anomalies_by_combined_method': anomaly_count_combined,
            'anomaly_rate_classification': (anomaly_count_class / len(predictions)) * 100,
            'anomaly_rate_combined': (anomaly_count_combined / len(predictions)) * 100,
            'reconstruction_threshold': recon_threshold,
            'mean_reconstruction_error': np.mean(reconstruction_errors),
            'std_reconstruction_error': np.std(reconstruction_errors),
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        print(f"\nPrediction Summary:")
        print(f"Total records: {summary['total_records']}")
        print(f"Anomalies (classification only): {summary['anomalies_by_classification']} ({summary['anomaly_rate_classification']:.1f}%)")
        print(f"Anomalies (combined method): {summary['anomalies_by_combined_method']} ({summary['anomaly_rate_combined']:.1f}%)")
        print(f"Mean reconstruction error: {summary['mean_reconstruction_error']:.4f}")
        
        # Detailed anomaly analysis
        if anomaly_count_combined > 0:
            anomalies_df = results_df[results_df['is_anomaly_combined'] == True]
            
            pattern1_count = sum(1 for p in predictions if p['is_anomaly_combined'] and p['predicted_class'] == 1)
            pattern2_count = sum(1 for p in predictions if p['is_anomaly_combined'] and p['predicted_class'] == 2)
            reconstruction_only = anomaly_count_combined - anomaly_count_class
            
            print(f"\nAnomaly Pattern Distribution:")
            print(f"  Pattern 1 (like dataset1): {pattern1_count}")
            print(f"  Pattern 2 (like dataset2): {pattern2_count}")
            print(f"  High reconstruction error only: {reconstruction_only}")
            
            # Show top anomalies
            top_anomalies = results_df.nlargest(min(5, anomaly_count_combined), 'anomaly_confidence')
            print(f"\nTop {len(top_anomalies)} Anomalies (by confidence):")
            for idx, row in top_anomalies.iterrows():
                print(f"  Record {row['node_index']}: Confidence={row['anomaly_confidence']:.3f}, "
                      f"Class={row['predicted_class']}, ReconError={row['reconstruction_error']:.4f}")
        
        # Save results
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            output_path = f"{base_name}_anomaly_predictions.csv"
        
        results_df.to_csv(output_path, index=False)
        
        # Save detailed report
        report_path = output_path.replace('.csv', '_report.json')
        detailed_report = {
            'summary': summary,
            'model_metadata': self.metadata,
            'input_file': csv_path,
            'output_file': output_path,
            'parameters': {
                'k_neighbors': k_neighbors,
                'threshold_multiplier': threshold_multiplier,
                'reconstruction_threshold': recon_threshold
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Detailed report saved to: {report_path}")
        
        return results_df, detailed_report

def main():
    parser = argparse.ArgumentParser(description='Deploy GNN Anomaly Detection Model')
    parser.add_argument('csv_file', help='Path to CSV file for anomaly detection')
    parser.add_argument('--model-path', default='output/final_gnn_anomaly_model.pth', 
                       help='Path to trained GNN model')
    parser.add_argument('--output', help='Output file path (optional)')
    parser.add_argument('--k-neighbors', type=int, default=5, 
                       help='Number of neighbors for graph construction')
    parser.add_argument('--threshold-multiplier', type=float, default=1.0, 
                       help='Multiplier for reconstruction error threshold')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found!")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        print("Please train the model first using: python train_gnn_model.py")
        return
    
    print(f"Input CSV: {args.csv_file}")
    print(f"Model path: {args.model_path}")
    print(f"K-neighbors: {args.k_neighbors}")
    print(f"Threshold multiplier: {args.threshold_multiplier}")
    
    try:
        # Initialize deployment engine
        engine = GNNDeploymentEngine(args.model_path)
        
        # Run anomaly detection
        results_df, report = engine.predict_anomalies(
            args.csv_file,
            args.output,
            args.threshold_multiplier,
            args.k_neighbors
        )
        
        print(f"\nDeployment completed successfully!")
        
        # Show quick stats
        anomaly_rate = report['summary']['anomaly_rate_combined']
        if anomaly_rate > 10:
            print(f"⚠️  High anomaly rate detected: {anomaly_rate:.1f}%")
        elif anomaly_rate > 5:
            print(f"⚡ Moderate anomaly rate: {anomaly_rate:.1f}%")
        else:
            print(f"✅ Normal anomaly rate: {anomaly_rate:.1f}%")
            
    except Exception as e:
        print(f"Error during deployment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()