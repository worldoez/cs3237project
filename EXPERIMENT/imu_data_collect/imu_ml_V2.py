import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# GPU check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Loading the data
df = pd.read_csv('all-combined_imu_data_randomized.csv')

# Clean data - convert to numeric and handle errors
feature_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']
for col in feature_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN values
df = df.dropna(subset=feature_columns)
print(f"Cleaned dataset size: {len(df)} rows")

# Check class distribution
print("\nClass distribution:")
print(df['action_label'].value_counts())

# Group by action_id into sequences
sequences = []
labels = []

for action_id in df['action_id'].unique():
    action_data = df[df['action_id'] == action_id]
    
    # Extract features for this sequence
    sequence = action_data[feature_columns].values
    sequences.append(sequence)
    
    # Get the label
    label = action_data['action_label'].iloc[0]
    labels.append(label)

print(f"\nTotal sequences: {len(sequences)}")
print(f"Sequence lengths - Min: {min(len(s) for s in sequences)}, Max: {max(len(s) for s in sequences)}")

# Convert labels to integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
print(f"Classes: {label_encoder.classes_}")

# Normalize the sequences
scaler = StandardScaler()
all_data = np.vstack(sequences)
scaler.fit(all_data)

sequences_normalized = []
for seq in sequences:
    seq_normalized = scaler.transform(seq)
    sequences_normalized.append(seq_normalized)

# USE TRUNCATION/PADDING to reasonable length
# Instead of using max_length, use a percentile to avoid extreme padding
seq_lengths = [len(seq) for seq in sequences_normalized]
max_length = int(np.percentile(seq_lengths, 95))  # Use 95th percentile
print(f"Using sequence length: {max_length} (95th percentile)")

def pad_or_truncate_sequence(seq, target_len):
    """Pad short sequences with zeros or truncate long ones"""
    if len(seq) > target_len:
        return seq[:target_len]  # Truncate
    else:
        padded = np.zeros((target_len, seq.shape[1]))
        padded[:len(seq)] = seq
        return padded

X = np.array([pad_or_truncate_sequence(seq, max_length) for seq in sequences_normalized])
y = np.array(labels_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Input shape: {X_train.shape[1:]} (timesteps, features)")

# Custom Dataset
class IMUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = IMUDataset(X_train, y_train)
test_dataset = IMUDataset(X_test, y_test)

# IMPROVED: Smaller batch size for better learning
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# IMPROVED LSTM Model
class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(ImprovedLSTMClassifier, self).__init__()
        
        # Bidirectional LSTM for better context
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism (simple)
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classifier
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Simple attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # Classification layers
        out = self.fc1(attended)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        return out

# Initialize improved model
num_classes = len(label_encoder.classes_)
model = ImprovedLSTMClassifier(
    input_size=6,
    hidden_size=128,  # Increased
    num_layers=2,
    num_classes=num_classes,
    dropout=0.3
).to(device)

print("\nModel Architecture:")
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer with weight decay
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Training loop with early stopping
num_epochs = 100
train_losses, train_accs = [], []
test_losses, test_accs = [], []
best_test_acc = 0
patience_counter = 0
patience = 15

print("\nTraining started...")
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # Learning rate scheduling
    scheduler.step(test_loss)
    
    # Early stopping
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        patience_counter = 0
        # Save best model
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'label_encoder': label_encoder,
            'max_length': max_length
        }, 'imu_lstm_model_best.pth')
    else:
        patience_counter += 1
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
              f"Best: {best_test_acc:.4f}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print("\nTraining completed!")
print(f"Best Test Accuracy: {best_test_acc * 100:.2f}%")
print(f"Final Test Accuracy: {test_accs[-1] * 100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.axhline(y=best_test_acc, color='r', linestyle='--', label=f'Best: {best_test_acc:.3f}')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("Training plot saved as 'training_history.png'")

# Load best model for final evaluation
checkpoint = torch.load('imu_lstm_model_best.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Detailed evaluation
def detailed_evaluation(model, dataloader, label_encoder, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_acc = (np.array(all_preds)[class_mask] == i).sum() / class_mask.sum()
            print(f"  {class_name}: {class_acc*100:.2f}% ({class_mask.sum()} samples)")

detailed_evaluation(model, test_loader, label_encoder, device)

# Example prediction
def predict_action(model, new_sequence, scaler, label_encoder, max_length, device):
    model.eval()
    seq_normalized = scaler.transform(new_sequence)
    seq_padded = pad_or_truncate_sequence(seq_normalized, max_length)
    seq_tensor = torch.FloatTensor(seq_padded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(seq_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
    
    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
    return predicted_label, confidence, probabilities[0].cpu().numpy()

# Test predictions
sample_idx = 0
sample_sequence = X_test[sample_idx]
actual_label = label_encoder.inverse_transform([y_test[sample_idx]])[0]

non_padded_length = np.sum(np.any(sample_sequence != 0, axis=1))
sample_for_prediction = sample_sequence[:non_padded_length]

predicted_label, confidence, all_probs = predict_action(
    model, sample_for_prediction, scaler, label_encoder, max_length, device
)
print(f"\nExample Prediction:")
print(f"Actual: {actual_label}")
print(f"Predicted: {predicted_label}")
print(f"Confidence: {confidence * 100:.2f}%")
print(f"All probabilities:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {class_name}: {all_probs[i]*100:.2f}%")