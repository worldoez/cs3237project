#!/usr/bin/env python3
"""
Test trained IMU LSTM model on new data or visualize predictions.

Usage:
    python test_imu_model.py test_data.csv
    python test_imu_model.py test_data.csv --visualize
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# Define the model architecture (must match training)
class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(ImprovedLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout * 0.7)
        
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        out = self.fc1(context)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        return out


def load_model(model_path='imu_lstm_model_best.pth', device='cpu'):
    """Load the trained model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model configuration from checkpoint
    input_size = 6  # gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
    hidden_size = 128
    num_layers = 2
    num_classes = len(checkpoint.get('classes', ['jump', 'left', 'right', 'straight']))
    
    model = ImprovedLSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    scaler = checkpoint.get('scaler', None)
    classes = checkpoint.get('classes', ['jump', 'left', 'right', 'straight'])
    seq_length = checkpoint.get('seq_length', 244)
    
    print(f"Model loaded successfully!")
    print(f"  Classes: {classes}")
    print(f"  Sequence length: {seq_length}")
    
    # Handle test_acc formatting
    test_acc = checkpoint.get('test_acc', None)
    if test_acc is not None:
        print(f"  Best test accuracy: {test_acc:.2%}")
    else:
        print(f"  Best test accuracy: N/A")
    
    return model, scaler, classes, seq_length


def prepare_sequences(df, seq_length, scaler=None):
    """Prepare sequences from dataframe."""
    feature_cols = ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']
    
    sequences = []
    labels = []
    action_ids = []
    
    for action_id in df['action_id'].unique():
        action_data = df[df['action_id'] == action_id]
        
        if len(action_data) < 10:
            continue
        
        features = action_data[feature_cols].values
        
        # Normalize
        if scaler is not None:
            features = scaler.transform(features)
        
        # Pad or truncate
        if len(features) < seq_length:
            pad_length = seq_length - len(features)
            features = np.vstack([features, np.zeros((pad_length, features.shape[1]))])
        else:
            features = features[:seq_length]
        
        sequences.append(features)
        labels.append(action_data['action_label'].iloc[0])
        action_ids.append(action_id)
    
    return np.array(sequences), np.array(labels), np.array(action_ids)


def test_model(model, test_file, scaler, classes, seq_length, device='cpu', visualize=False):
    """Test the model on new data."""
    # Load test data
    df = pd.read_csv(test_file)
    print(f"\nLoaded test data: {test_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique actions: {df['action_id'].nunique()}")
    
    # Prepare sequences
    X, y_true, action_ids = prepare_sequences(df, seq_length, scaler)
    print(f"\nPrepared {len(X)} sequences for testing")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
    
    # Map predictions back to class names
    y_pred = [classes[p] for p in predicted]
    
    # Calculate accuracy
    accuracy = np.mean(np.array(y_pred) == y_true)
    
    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Show sample predictions
    print(f"\nSample Predictions (first 10):")
    print(f"{'Action ID':<12} {'True Label':<12} {'Predicted':<12} {'Confidence':<12} {'Correct'}")
    print("-" * 60)
    for i in range(min(10, len(y_pred))):
        confidence = probabilities[i][predicted[i]]
        correct = "✓" if y_pred[i] == y_true[i] else "✗"
        print(f"{action_ids[i]:<12} {y_true[i]:<12} {y_pred[i]:<12} {confidence:<12.2%} {correct}")
    
    if len(y_pred) > 10:
        print(f"... and {len(y_pred) - 10} more predictions")
    
    # Visualize if requested
    if visualize:
        visualize_results(y_true, y_pred, classes, probabilities, predicted)
    
    return y_true, y_pred, probabilities


def visualize_results(y_true, y_pred, classes, probabilities, predicted):
    """Create visualization of test results."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    im = axes[0].imshow(cm, cmap='Blues', aspect='auto')
    axes[0].figure.colorbar(im, ax=axes[0])
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', 
                        color=text_color, fontsize=12, weight='bold')
    
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_xticklabels(classes)
    axes[0].set_yticklabels(classes)
    axes[0].set_title('Confusion Matrix', fontsize=14, weight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Confidence distribution
    confidences = [probabilities[i][predicted[i]] for i in range(len(predicted))]
    correct = np.array(y_pred) == np.array(y_true)
    
    axes[1].hist([np.array(confidences)[correct], np.array(confidences)[~correct]], 
                 bins=20, label=['Correct', 'Incorrect'], alpha=0.7, 
                 color=['green', 'red'])
    axes[1].set_title('Prediction Confidence Distribution', fontsize=14, weight='bold')
    axes[1].set_xlabel('Confidence', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as 'test_results.png'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test trained IMU LSTM model')
    parser.add_argument('test_file', help='CSV file with test data')
    parser.add_argument('--model', default='imu_lstm_model_best.pth', 
                       help='Path to trained model (default: imu_lstm_model_best.pth)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.test_file).exists():
        print(f"Error: Test file not found: {args.test_file}")
        return
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, scaler, classes, seq_length = load_model(args.model, device)
    
    # Test model
    test_model(model, args.test_file, scaler, classes, seq_length, device, args.visualize)


if __name__ == '__main__':
    main()