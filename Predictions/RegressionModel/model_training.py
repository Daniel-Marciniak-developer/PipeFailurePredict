import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import os

class FailureDataset(Dataset):
    """
    PyTorch Dataset for pipe failure data.

    Attributes:
        X (Tensor): Input features.
        y (Tensor): Target values.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RegressionModel(nn.Module):
    """
    Neural network model for regression tasks.

    Attributes:
        network (nn.Sequential): The neural network architecture.
    """
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

class PipeFailureModelTrainer:
    """
    Class responsible for training the pipe failure prediction model.

    Methods:
        load_data(): Loads and preprocesses the data.
        train_model(): Trains the neural network model.
        save_model(): Saves the trained model and preprocessing objects.
    """
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

    def load_data(self):
        """
        Loads data from CSV file and preprocesses it.

        Returns:
            df (DataFrame): Preprocessed data.
            reference_date (Timestamp): Reference date for date calculations.
        """
        df = pd.read_csv(self.data_path)
        df.drop(df.columns[[6, 8, 9, 10]], axis=1, inplace=True)
        df['Data_pomiaru'] = pd.to_datetime(df['Data_pomiaru'], errors='coerce')
        df['Data_ostatniej_awarii'] = pd.to_datetime(df['Data_ostatniej_awarii'], errors='coerce')
        df['Data_prawdziwej_awarii'] = pd.to_datetime(df['Data_prawdziwej_awarii'], errors='coerce')
        df.dropna(subset=['Data_pomiaru', 'Data_ostatniej_awarii', 'Data_prawdziwej_awarii',
                          'Materiał', 'Wiek_rury', 'Średnica', 'Długość_odcinka', 'Przepływ'], inplace=True)
        reference_date = min(df['Data_pomiaru'].min(), df['Data_ostatniej_awarii'].min()) - pd.Timedelta(days=1)
        df['days_since_measurement'] = (df['Data_pomiaru'] - reference_date).dt.days
        df['days_since_last_failure'] = (df['Data_ostatniej_awarii'] - reference_date).dt.days
        df['days_until_true_failure'] = (df['Data_prawdziwej_awarii'] - df['Data_pomiaru']).dt.days
        df['days_until_true_failure'].fillna(-1, inplace=True)
        df.loc[df['days_until_true_failure'] < 0, 'days_until_true_failure'] = -1
        df['age_diameter_ratio'] = df['Wiek_rury'] / (df['Średnica'] + 1)
        df.dropna(subset=['days_since_measurement', 'days_since_last_failure', 'age_diameter_ratio'], inplace=True)
        return df, reference_date

    def encode_features(self, df, categorical_columns, numerical_columns):
        """
        Encodes categorical features and scales numerical features.

        Args:
            df (DataFrame): Input data.
            categorical_columns (list): List of categorical column names.
            numerical_columns (list): List of numerical column names.

        Returns:
            features (ndarray): Encoded and scaled features.
            feature_names (list): Names of all features.
            encoder (OneHotEncoder): Fitted encoder.
            scaler (StandardScaler): Fitted scaler.
        """
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_categories = encoder.fit_transform(df[categorical_columns])
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[numerical_columns])
        features = np.concatenate([scaled_features, encoded_categories], axis=1)
        feature_names = numerical_columns + list(encoder.get_feature_names_out(categorical_columns))
        return features, feature_names, encoder, scaler

    def extract_constant_features(self, df, constant_columns):
        """
        Extracts constant features for each pipeline segment.

        Args:
            df (DataFrame): Input data.
            constant_columns (list): List of constant column names.

        Returns:
            context_features (DataFrame): Extracted context features.
        """
        return df.groupby('ID_odcinka', as_index=False)[constant_columns].first()

    def encode_constant_features(self, context_features, categorical_columns, numerical_columns):
        """
        Encodes and scales constant context features.

        Args:
            context_features (DataFrame): Contextual features.
            categorical_columns (list): List of categorical column names.
            numerical_columns (list): List of numerical column names.

        Returns:
            context_vectors (ndarray): Encoded and scaled context features.
            context_feature_names (list): Names of context features.
            encoder_constant (OneHotEncoder): Fitted encoder for context features.
            scaler_constant (StandardScaler): Fitted scaler for context features.
        """
        context_features_for_encoding = context_features.drop(columns=['ID_odcinka'])
        encoder_constant = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_categories = encoder_constant.fit_transform(context_features_for_encoding[categorical_columns])
        scaler_constant = StandardScaler()
        scaled_numerical = scaler_constant.fit_transform(context_features_for_encoding[numerical_columns])
        context_vectors = np.concatenate([scaled_numerical, encoded_categories], axis=1)
        context_feature_names = numerical_columns + list(encoder_constant.get_feature_names_out(categorical_columns))
        return context_vectors, context_feature_names, encoder_constant, scaler_constant

    def create_features_dataframe(self, df, features, context_vectors, context_features):
        """
        Creates a DataFrame combining features and context vectors.

        Args:
            df (DataFrame): Original data.
            features (ndarray): Feature array.
            context_vectors (ndarray): Contextual feature array.
            context_features (DataFrame): Context features DataFrame.

        Returns:
            df_combined (DataFrame): Combined DataFrame.
        """
        df_combined = df.copy()
        context_dict = dict(zip(context_features['ID_odcinka'], context_vectors))
        df_combined['context'] = df_combined['ID_odcinka'].map(context_dict)
        df_combined['features'] = list(features)
        df_combined = df_combined[df_combined['days_until_true_failure'] != -1]
        df_combined.dropna(subset=['features', 'context'], inplace=True)
        return df_combined

    def save_model(self, model, scaler_X, encoder_X, scaler_y, encoder_constant, scaler_constant,
                   feature_names, context_feature_names):
        """
        Saves the trained model and preprocessing objects.

        Args:
            model (nn.Module): Trained model.
            scaler_X (StandardScaler): Scaler for features.
            encoder_X (OneHotEncoder): Encoder for features.
            scaler_y (StandardScaler): Scaler for target variable.
            encoder_constant (OneHotEncoder): Encoder for context features.
            scaler_constant (StandardScaler): Scaler for context features.
            feature_names (list): Names of features.
            context_feature_names (list): Names of context features.
        """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(model.state_dict(), os.path.join(self.model_path, 'regression_model.pth'))
        joblib.dump(scaler_X, os.path.join(self.model_path, 'scaler_X.pkl'))
        joblib.dump(encoder_X, os.path.join(self.model_path, 'encoder_X.pkl'))
        joblib.dump(scaler_y, os.path.join(self.model_path, 'scaler_y.pkl'))
        joblib.dump(encoder_constant, os.path.join(self.model_path, 'encoder_constant.pkl'))
        joblib.dump(scaler_constant, os.path.join(self.model_path, 'scaler_constant.pkl'))
        joblib.dump(feature_names, os.path.join(self.model_path, 'feature_names.pkl'))
        joblib.dump(context_feature_names, os.path.join(self.model_path, 'context_feature_names.pkl'))

    def train_model(self):
        """
        Trains the neural network model.
        """
        df, reference_date = self.load_data()
        categorical_columns = ['Materiał']
        numerical_columns = ['Długość_odcinka', 'Wiek_rury', 'Średnica', 'Przepływ',
                             'days_since_last_failure', 'age_diameter_ratio']
        features, feature_names, encoder_X, scaler_X = self.encode_features(df, categorical_columns, numerical_columns)
        constant_columns = ['ID_odcinka'] + categorical_columns + ['Średnica', 'Wiek_rury', 'Długość_odcinka']
        context_features = self.extract_constant_features(df, constant_columns)
        context_vectors, context_feature_names, encoder_constant, scaler_constant = self.encode_constant_features(
            context_features, categorical_columns, ['Średnica', 'Wiek_rury', 'Długość_odcinka']
        )
        df_features = self.create_features_dataframe(df, features, context_vectors, context_features)
        X = np.vstack(df_features['features'].values)
        context_X = np.vstack(df_features['context'].values)
        X_final = np.hstack([X, context_X])
        y = df_features['days_until_true_failure'].values
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_scaled, test_size=0.2, random_state=42)
        train_dataset = FailureDataset(X_train, y_train)
        test_dataset = FailureDataset(X_test, y_test)
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        input_size = X_final.shape[1]
        model = RegressionModel(input_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        epochs = 50
        best_mse = float('inf')
        patience = 5
        trigger_times = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_batch.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            epoch_val_loss = val_loss / len(test_loader.dataset)
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')
            if epoch_val_loss < best_mse:
                best_mse = epoch_val_loss
                trigger_times = 0
                self.save_model(model, scaler_X, encoder_X, scaler_y, encoder_constant, scaler_constant,
                                feature_names, context_feature_names)
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print('Early stopping!')
                    break
        print('Training completed.')

def main():
    data_path = 'PipeFailurePredict--WaterPrime\\Predictions\\data\\data_failure.csv'
    model_path = 'D:\Projects\Aiut\PipeFailurePredict--WaterPrime\Predictions\RegressionModel\models'
    trainer = PipeFailureModelTrainer(data_path, model_path)
    trainer.train_model()

if __name__ == "__main__":
    main()
