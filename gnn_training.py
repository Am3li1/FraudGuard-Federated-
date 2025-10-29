# train_gnn_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import NearestNeighbors
import json
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import the GraphConstructor from detection script
from anomaly_gnn_2 import GraphConstructor

# Import required libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

class NodeLevelGNNAnomalyDetector(nn.Module):
    """Graph Neural Network for node-level anomaly detection"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, num_layers=3, dropout=0.3):
        super(NodeLevelGNNAnomalyDetector, self).__init__()
        
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
        
        # Node-level classification head
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
        
        # Node-level classification output
        classification_out = self.classifier(h)
        
        # Reconstruction output
        reconstruction_out = self.reconstructor(h)
        
        return classification_out, reconstruction_out, h

class MainGNNTrainer:
    """Main GNN model trainer using anomaly reports"""
    
    def __init__(self, input_dim, hidden_dim=128, num_classes=3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.model = NodeLevelGNNAnomalyDetector(input_dim, hidden_dim, num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        self.criterion_classification = nn.CrossEntropyLoss()
        self.criterion_reconstruction = nn.MSELoss()
        self.training_history = {'loss': [], 'accuracy': []}
        
    def prepare_training_data(self, training_data_file, test_split=0.2, val_split=0.1):
        """Prepare training data from labeled dataset"""
        
        df = pd.read_csv(training_data_file)
        print(f"Loaded training data with shape: {df.shape}")
        
        # Check class distribution
        class_counts = df['anomaly_type'].value_counts().sort_index()
        print("Class distribution:")
        for class_id, count in class_counts.items():
            class_name = {0: "Normal", 1: "Anomaly Pattern 1", 2: "Anomaly Pattern 2"}.get(class_id, f"Class {class_id}")
            print(f"  {class_name}: {count} samples ({count/len(df)*100:.1f}%)")
        
        # Prepare graph data
        graph_constructor = GraphConstructor(k_neighbors=5)
        
        # Remove non-feature columns
        feature_columns = [col for col in df.columns if col not in ['anomaly_type', 'dataset_source']]
        
        # Handle categorical variables
        categorical_columns = df[feature_columns].select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        # Handle missing values
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
        
        # Extract features and labels
        X = df[feature_columns].values
        y = df['anomaly_type'].values
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_split, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_split/(1-test_split), stratify=y_temp, random_state=42)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples") 
        print(f"Test set: {len(X_test)} samples")
        
        # Create graphs for each split
        def create_graph_data(features, labels):
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Create k-NN graph
            nbrs = NearestNeighbors(n_neighbors=min(6, len(features)), algorithm='ball_tree')
            nbrs.fit(features_scaled)
            distances, indices = nbrs.kneighbors(features_scaled)
            
            edge_list = []
            edge_weights = []
            
            for i, neighbors in enumerate(indices):
                for j, neighbor_idx in enumerate(neighbors[1:]):  # Skip self
                    if j < len(neighbors) - 1:  # Safety check
                        edge_list.append([i, neighbor_idx])
                        edge_weights.append(1.0 / (distances[i][j+1] + 1e-8))
            
            if len(edge_list) == 0:
                # Fallback: create chain connections
                for i in range(min(len(features) - 1, 100)):
                    edge_list.append([i, i+1])
                    edge_weights.append(1.0)
            
            edge_index = torch.LongTensor(np.array(edge_list).T)
            edge_attr = torch.FloatTensor(edge_weights)
            x = torch.FloatTensor(features_scaled)
            y_tensor = torch.LongTensor(labels)  # Keep node-level labels
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor), scaler
        
        train_data, self.scaler = create_graph_data(X_train, y_train)
        val_data, _ = create_graph_data(X_val, y_val)
        test_data, _ = create_graph_data(X_test, y_test)
        
        return train_data, val_data, test_data, label_encoders
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in data_loader:
            self.optimizer.zero_grad()
            
            classification_out, reconstruction_out, _ = self.model(batch)
            
            # Node-level classification loss
            class_loss = self.criterion_classification(classification_out, batch.y)
            
            # Reconstruction loss
            recon_loss = self.criterion_reconstruction(reconstruction_out, batch.x)
            
            # Combined loss with reconstruction regularization
            total_loss_batch = class_loss + 0.1 * recon_loss
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # Calculate accuracy
            predicted = torch.argmax(classification_out, dim=1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
        
        return total_loss / len(data_loader), correct / total
    
    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                classification_out, reconstruction_out, _ = self.model(batch)
                
                # Node-level evaluation
                class_loss = self.criterion_classification(classification_out, batch.y)
                recon_loss = self.criterion_reconstruction(reconstruction_out, batch.x)
                total_loss_batch = class_loss + 0.1 * recon_loss
                
                total_loss += total_loss_batch.item()
                
                predicted = torch.argmax(classification_out, dim=1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        return total_loss / len(data_loader), correct / total, all_predictions, all_labels
    
    def train(self, train_data, val_data, epochs=100, patience=15, output_dir="output"):
        """Train the main GNN model"""
        
        # Create data loaders
        train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
        val_loader = DataLoader([val_data], batch_size=1, shuffle=False)
        
        best_val_acc = 0
        patience_counter = 0
        best_model_state = None
        
        print(f"Starting training for {epochs} epochs...")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.evaluate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save training history
            self.training_history['loss'].append({'train': train_loss, 'val': val_loss})
            self.training_history['accuracy'].append({'train': train_acc, 'val': val_acc})
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
        
        # Save training plots
        self.plot_training_history(output_dir)
        
        return best_val_acc
    
    def plot_training_history(self, output_dir):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        epochs = range(len(self.training_history['loss']))
        train_losses = [h['train'] for h in self.training_history['loss']]
        val_losses = [h['val'] for h in self.training_history['loss']]
        
        ax1.plot(epochs, train_losses, label='Training Loss')
        ax1.plot(epochs, val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        train_accs = [h['train'] for h in self.training_history['accuracy']]
        val_accs = [h['val'] for h in self.training_history['accuracy']]
        
        ax2.plot(epochs, train_accs, label='Training Accuracy')
        ax2.plot(epochs, val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to {output_dir}/training_history.png")
    
    def evaluate_test_set(self, test_data, output_dir):
        """Evaluate on test set and generate detailed report"""
        test_loader = DataLoader([test_data], batch_size=1, shuffle=False)
        
        test_loss, test_acc, test_preds, test_labels = self.evaluate(test_loader)
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Check if we have multiple classes in test set
        unique_test_labels = np.unique(test_labels)
        unique_test_preds = np.unique(test_preds)
        
        print(f"Test labels present: {unique_test_labels}")
        print(f"Predictions made: {unique_test_preds}")
        
        # Classification report with proper label handling
        class_names = ['Normal', 'Anomaly Pattern 1', 'Anomaly Pattern 2']
        
        # Only use labels that appear in the test set or predictions
        all_labels = sorted(list(set(test_labels + test_preds)))
        used_class_names = [class_names[i] if i < len(class_names) else f'Class {i}' for i in all_labels]
        
        try:
            report = classification_report(test_labels, test_preds, 
                                         labels=all_labels,
                                         target_names=used_class_names, 
                                         output_dict=True,
                                         zero_division=0)
            
            print("\nDetailed Classification Report:")
            print(classification_report(test_labels, test_preds, 
                                      labels=all_labels,
                                      target_names=used_class_names,
                                      zero_division=0))
        except Exception as e:
            print(f"Could not generate full classification report: {e}")
            report = {"accuracy": test_acc}
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, test_preds, labels=all_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=used_class_names, yticklabels=used_class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
        
        # Save detailed results
        results = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'unique_labels': all_labels,
            'class_distribution': {
                'test_labels': {str(label): int(np.sum(np.array(test_labels) == label)) for label in all_labels},
                'predictions': {str(label): int(np.sum(np.array(test_preds) == label)) for label in all_labels}
            }
        }
        
        results_file = os.path.join(output_dir, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Detailed test results saved to {results_file}")
        
        return results
    
    def save_model(self, filepath, metadata=None):
        """Save the trained model with metadata"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_class': 'NodeLevelGNNAnomalyDetector',  # Specify the model class
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            save_dict.update(metadata)
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create model with saved parameters
        input_dim = checkpoint.get('input_dim', self.input_dim)
        hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
        num_classes = checkpoint.get('num_classes', self.num_classes)
        
        # Reinitialize model if dimensions changed
        if (input_dim != self.input_dim or hidden_dim != self.hidden_dim or 
            num_classes != self.num_classes):
            self.model = NodeLevelGNNAnomalyDetector(input_dim, hidden_dim, num_classes)
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
            
        print(f"Model loaded from {filepath}")
        return checkpoint

def main_training_workflow(training_data_file, metadata_file, output_dir="output", epochs=100, test_split=0.2, val_split=0.1):
    """Complete training workflow"""
    
    print("=== GNN Model Training Workflow ===")
    
    # Load metadata
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {metadata_file}")
        input_dim = metadata['input_dim']
    else:
        print("Warning: No metadata file found, will infer input dimension from data")
        df = pd.read_csv(training_data_file)
        feature_columns = [col for col in df.columns if col not in ['anomaly_type', 'dataset_source']]
        input_dim = len(feature_columns)
    
    print(f"Input dimension: {input_dim}")
    
    # Initialize trainer
    trainer = MainGNNTrainer(input_dim, hidden_dim=128, num_classes=3)
    
    # Prepare training data
    print("\nPreparing training data...")
    train_data, val_data, test_data, label_encoders = trainer.prepare_training_data(
        training_data_file, test_split=test_split, val_split=val_split
    )
    
    # Train the model
    print(f"\nStarting training...")
    best_val_acc = trainer.train(train_data, val_data, epochs=epochs, output_dir=output_dir)
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_results = trainer.evaluate_test_set(test_data, output_dir)
    
    # Save the final model
    model_path = os.path.join(output_dir, "final_gnn_anomaly_model.pth")
    training_metadata = {
        'best_validation_accuracy': best_val_acc,
        'test_accuracy': test_results['test_accuracy'],
        'label_encoders': {k: list(v.classes_) for k, v in label_encoders.items()},
        'training_data_file': training_data_file
    }
    trainer.save_model(model_path, training_metadata)
    
    print("\n=== Training Complete ===")
    print("Files generated:")
    print(f"1. {model_path} - Trained GNN model")
    print(f"2. {os.path.join(output_dir, 'training_history.png')} - Training plots")
    print(f"3. {os.path.join(output_dir, 'confusion_matrix.png')} - Confusion matrix")
    print(f"4. {os.path.join(output_dir, 'test_results.json')} - Detailed test results")
    print(f"\nFinal test accuracy: {test_results['test_accuracy']:.4f}")
    
    return trainer, test_results

def main():
    parser = argparse.ArgumentParser(description='GNN Anomaly Detection - Training Phase')
    parser.add_argument('--training-data', default='output/training_data_with_labels.csv', 
                       help='Path to training data CSV file')
    parser.add_argument('--metadata', default='output/training_metadata.json', 
                       help='Path to training metadata JSON file')
    parser.add_argument('--output-dir', default='output', 
                       help='Output directory for training results')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--test-split', type=float, default=0.2, 
                       help='Test set split ratio')
    parser.add_argument('--val-split', type=float, default=0.1, 
                       help='Validation set split ratio')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.training_data):
        print(f"Error: Training data file '{args.training_data}' not found!")
        print("Please run the detection phase first: python anomaly_detection.py <dataset1> <dataset2>")
        return
    
    print(f"Training data: {args.training_data}")
    print(f"Metadata file: {args.metadata}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Test split: {args.test_split}")
    print(f"Validation split: {args.val_split}")
    
    # Run training workflow
    try:
        trainer, results = main_training_workflow(
            args.training_data,
            args.metadata,
            args.output_dir,
            args.epochs,
            args.test_split,
            args.val_split
        )
        
        print("\nTraining completed successfully!")
        print(f"Model is ready for deployment at {args.output_dir}/final_gnn_anomaly_model.pth")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()