import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(df.columns[[6, 8, 9, 10]], axis=1, inplace=True)
    df['Data_pomiaru'] = pd.to_datetime(df['Data_pomiaru'], errors='coerce')
    df['Data_ostatniej_awarii'] = pd.to_datetime(df['Data_ostatniej_awarii'], errors='coerce')
    df['Data_prawdziwej_awarii'] = pd.to_datetime(df['Data_prawdziwej_awarii'], errors='coerce')
    return df

def preprocess_data(df):
    reference_date = min(df['Data_pomiaru'].min(), df['Data_ostatniej_awarii'].min()) - pd.Timedelta(days=1)
    df['days_since_measurement'] = (df['Data_pomiaru'] - reference_date).dt.days
    df['days_since_last_failure'] = (df['Data_ostatniej_awarii'] - reference_date).dt.days
    df['days_until_true_failure'] = (df['Data_prawdziwej_awarii'] - reference_date).dt.days
    df['days_until_true_failure'].fillna(-1, inplace=True)
    return df, reference_date

def encode_features(df, categorical_columns, numerical_columns):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categories = encoder.fit_transform(df[categorical_columns])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numerical_columns])
    features = np.concatenate([scaled_features, encoded_categories], axis=1)
    return features, encoder, scaler

def extract_constant_features(df, constant_columns):
    return df.groupby('ID_odcinka', as_index=False)[constant_columns].first()

def encode_constant_features(context_features, categorical_columns, numerical_columns, df):
    context_features_for_encoding = context_features.drop(columns=['ID_odcinka'])
    encoder_constant = OneHotEncoder(sparse_output=False).fit(df[categorical_columns])
    context_encoded = encoder_constant.transform(context_features_for_encoding[categorical_columns])
    context_scaler = StandardScaler().fit(df[numerical_columns])
    context_scaled = context_scaler.transform(context_features_for_encoding[numerical_columns])
    context_vectors = np.concatenate([context_scaled, context_encoded], axis=1)
    return context_vectors, encoder_constant, context_scaler

def create_sequences(df, features, context_vectors, context_features, reference_date):
    X_seq, y_seq, context_seq = [], [], []

    for segment_id in df['ID_odcinka'].unique():
        segment_data = df[df['ID_odcinka'] == segment_id]
        sequence_length = min(365, len(segment_data))

        if len(segment_data) < sequence_length:
            continue

        segment_indices = segment_data.index[:sequence_length]
        segment_features = features[segment_indices]

        if segment_id in context_features['ID_odcinka'].values:
            context_index = context_features[context_features['ID_odcinka'] == segment_id].index[0]
            context_vector = context_vectors[context_index]
        else:
            continue

        future_failure_date = segment_data['days_until_true_failure'].iloc[-1]
        if pd.isna(future_failure_date) or future_failure_date == -1:
            continue

        X_seq.append(segment_features)
        context_seq.append(context_vector)
        y_seq.append(future_failure_date)

    return X_seq, context_seq, y_seq

class LSTMModel(nn.Module):
    """
    LSTM-based model for predicting pipeline failures.
    """
    def __init__(self, input_size, context_size, hidden_size, num_layers, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.context_embedding = nn.Linear(context_size, hidden_size)  # Embed context to match LSTM hidden size
        self.input_embedding = nn.Linear(input_size, hidden_size)  # Transform input to match context size
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x, context):
        # Transform the input features to match hidden size
        x = self.input_embedding(x)

        # Embed the context and repeat it along the sequence length to match x's shape
        context = self.context_embedding(context).unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Debugging the shapes before adding
        print(f"x shape after embedding: {x.shape}, context shape: {context.shape}")
        
        # Ensure x and context are compatible
        x = x + context  # Element-wise addition between sequences and context vectors
        lstm_out, _ = self.lstm(x)
        
        # Return the prediction based on the last output of the LSTM
        return self.fc_out(lstm_out[:, -1, :])

def custom_loss(y_pred, y_true):
    """
    Custom loss function with penalties for incorrect predictions.
    """
    loss = torch.nn.MSELoss()(y_pred, y_true)
    penalty_late = torch.mean(torch.relu(y_pred - y_true))
    penalty_early = torch.mean(torch.relu(30 - (y_true - y_pred)))
    return loss + penalty_late + penalty_early

def train_model(model, optimizer, scheduler, X_train_seq, y_train_seq, context_train_seq, X_val_seq, y_val_seq, context_val_seq, num_epochs):
    """
    Trains the model for a specified number of epochs with validation and early stopping.
    """
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_seq, context_train_seq).squeeze()
        loss = custom_loss(outputs, y_train_seq)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_seq, context_val_seq).squeeze()
            val_loss = custom_loss(val_outputs, y_val_seq)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        scheduler.step(val_loss)

        # Save the best model
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            torch.save(model.state_dict(), 'best_model.pth')

def evaluate_model(model, X_test_seq, context_test_seq, y_test_seq, y_scaler, reference_date):
    """
    Evaluates the model and visualizes the results.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_seq, context_test_seq).cpu().numpy().flatten()
        predicted_days = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten().astype(int)
        predicted_dates = [reference_date + pd.to_timedelta(day, unit='D') for day in predicted_days]
        actual_days = y_scaler.inverse_transform(y_test_seq.cpu().numpy().reshape(-1, 1)).flatten().astype(int)
        actual_dates = [reference_date + pd.to_timedelta(day, unit='D') for day in actual_days]

    visualization_data = pd.DataFrame({
        'Predicted Failure Date': predicted_dates,
        'Actual Failure Date': actual_dates
    })
    print("\nSample predictions vs actual failure dates:")
    print(visualization_data.head(40))

    plt.figure(figsize=(12, 6))
    plt.plot(predicted_dates[:40], label='Predicted Failure Dates', marker='o', linestyle='--')
    plt.plot(actual_dates[:40], label='Actual Failure Dates', marker='x', linestyle='-')
    plt.xlabel('Samples')
    plt.ylabel('Failure Dates')
    plt.legend()
    plt.title('Comparison of Predicted and Actual Failure Dates')
    plt.show()

# Load and preprocess the data
data_path = 'data_failure.csv'
df = load_data(data_path)
df, reference_date = preprocess_data(df)

categorical_columns = ['Materiał']
numerical_columns = ['Długość_odcinka', 'Wiek_rury', 'Średnica', 'Przepływ', 'days_since_measurement', 'days_since_last_failure']
features, encoder, scaler = encode_features(df, categorical_columns, numerical_columns)

constant_columns = ['ID_odcinka'] + categorical_columns + ['Średnica', 'Wiek_rury', 'Długość_odcinka']
context_features = extract_constant_features(df, constant_columns)
context_vectors, encoder_constant, context_scaler = encode_constant_features(context_features, categorical_columns, ['Średnica', 'Wiek_rury', 'Długość_odcinka'], df)

X_seq, context_seq, y_seq = create_sequences(df, features, context_vectors, context_features, reference_date)
X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32).to(device)
context_seq = torch.tensor(np.array(context_seq), dtype=torch.float32).to(device)
y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32).to(device)

# Normalize target
y_scaler = StandardScaler()
# Normalize target
y_scaler = StandardScaler()
y_seq_scaled = y_scaler.fit_transform(y_seq.cpu().numpy().reshape(-1, 1)).flatten()
y_seq = torch.tensor(y_seq_scaled, dtype=torch.float32).to(device)

# Split into train, validation, and test sets
X_train_seq, X_val_test_seq, y_train_seq, y_val_test_seq, context_train_seq, context_val_test_seq = train_test_split(
    X_seq, y_seq, context_seq, test_size=0.3, random_state=42)
X_val_seq, X_test_seq, y_val_seq, y_test_seq, context_val_seq, context_test_seq = train_test_split(
    X_val_test_seq, y_val_test_seq, context_val_test_seq, test_size=0.5, random_state=42)

# Model initialization and training
input_size = X_train_seq.shape[2]
context_size = context_train_seq.shape[1]
hidden_size = 128
num_layers = 2
model = LSTMModel(input_size, context_size, hidden_size, num_layers, dropout=0.2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

num_epochs = 100
train_model(model, optimizer, scheduler, X_train_seq, y_train_seq, context_train_seq, X_val_seq, y_val_seq, context_val_seq, num_epochs)

# Evaluation
evaluate_model(model, X_test_seq, context_test_seq, y_test_seq, y_scaler, reference_date)

