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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# Loading the data
df = pd.read_csv('EXAMPLE-imu_data_20251017_181344.csv')

# Separate features
feature_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']

# Group by action_id into sequences --- Cleaning the CSV data
sequences = []
labels = []

for action_id in df['action_id'].unique():
    action_data = df[df['action_id'] == action_id]
    
    # Extract features for this sequence
    sequence = action_data[feature_columns].values
    sequences.append(sequence)
    
    # Get the label (same for all rows in this action)
    label = action_data['action_label'].iloc[0]
    labels.append(label)

# Convert labels to integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
print(f"Classes: {label_encoder.classes_}")

# Normalize the sequences
scaler = StandardScaler()

# Flatten all sequences to fit the scaler
all_data = np.vstack(sequences)
scaler.fit(all_data)

# Normalize each sequence
sequences_normalized = []
for seq in sequences:
    seq_normalized = scaler.transform(seq)
    sequences_normalized.append(seq_normalized)

# Pad sequences to the same length
max_length = max([len(seq) for seq in sequences_normalized])
print(f"Max sequence length: {max_length}")

# Manual padding function
def pad_sequence(seq, max_len):
    padded = np.zeros((max_len, seq.shape[1]))
    padded[:len(seq)] = seq
    return padded

X = np.array([pad_sequence(seq, max_length) for seq in sequences_normalized])
y = np.array(labels_encoded)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Input shape: {X_train.shape[1:]} (timesteps, features)")

# Custom Dataset class
class IMUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets and dataloaders
train_dataset = IMUDataset(X_train, y_train)
test_dataset = IMUDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # LSTM layer 1
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        
        # LSTM layer 2
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Take the output from the last time step
        last_output = lstm_out2[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out

# Initialize model
num_classes = len(label_encoder.classes_)
model = LSTMClassifier(
    input_size=6,
    hidden_size1=64,
    hidden_size2=32,
    num_classes=num_classes,
    dropout=0.3
).to(device)

print("\nModel Architecture:")
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

# Training loop
num_epochs = 50
train_losses, train_accs = [], []
test_losses, test_accs = [], []

print("\nTraining started...")
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

print("\nTraining completed!")
print(f"Final Test Accuracy: {test_accs[-1] * 100:.2f}%")
print(f"Final Test Loss: {test_losses[-1]:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
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
plt.show()

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'label_encoder': label_encoder,
    'max_length': max_length
}, 'imu_lstm_model.pth')
print("\nModel saved as 'imu_lstm_model.pth'")

# Function to predict new data
def predict_action(model, new_sequence, scaler, label_encoder, max_length, device):
    """
    Predict action from a new IMU sequence
    
    Parameters:
    new_sequence: numpy array of shape (timesteps, 6)
    """
    model.eval()
    
    # Normalize
    seq_normalized = scaler.transform(new_sequence)
    
    # Pad to max_length
    seq_padded = pad_sequence(seq_normalized, max_length)
    
    # Convert to tensor
    seq_tensor = torch.FloatTensor(seq_padded).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(seq_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
    
    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
    
    return predicted_label, confidence

# Example prediction on test data
sample_idx = 0
sample_sequence = X_test[sample_idx]
actual_label = label_encoder.inverse_transform([y_test[sample_idx]])[0]

# Remove padding for visualization
non_padded_length = np.sum(np.any(sample_sequence != 0, axis=1))
sample_for_prediction = sample_sequence[:non_padded_length]

predicted_label, confidence = predict_action(
    model, sample_for_prediction, scaler, label_encoder, max_length, device
)
print(f"\nExample Prediction:")
print(f"Actual: {actual_label}")
print(f"Predicted: {predicted_label}")
print(f"Confidence: {confidence * 100:.2f}%")