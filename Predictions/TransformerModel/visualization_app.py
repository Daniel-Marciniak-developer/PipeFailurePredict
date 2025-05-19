import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Definicja Modelu Transformer (taka sama jak w skrypcie trenowania)
class TransformerModel(nn.Module):
    """
    Model Transformer do przewidywania awarii rur.
    """
    def __init__(self, input_size, context_size, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.context_embedding = nn.Linear(context_size, d_model)
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x, context):
        x = self.embedding(x)
        context = self.context_embedding(context).unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + context
        x = self.transformer_encoder(x)
        return self.fc_out(x[:, -1, :])

def load_model_and_preprocessors(model_path):
    """
    Ładuje wytrenowany model oraz obiekty przetwarzania danych.

    Args:
        model_path (str): Ścieżka do katalogu z modelem i obiektami przetwarzania danych.

    Returns:
        model (nn.Module): Załadowany model.
        encoder (OneHotEncoder): Enkoder dla cech kategorycznych.
        scaler (StandardScaler): Skalowacz dla cech numerycznych.
        encoder_constant (OneHotEncoder): Enkoder dla cech kontekstowych.
        context_scaler (StandardScaler): Skalowacz dla cech kontekstowych.
        y_scaler (StandardScaler): Skalowacz dla zmiennej celu.
    """
    try:
        encoder = joblib.load(os.path.join(model_path, 'encoder.pkl'))
        scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        encoder_constant = joblib.load(os.path.join(model_path, 'encoder_constant.pkl'))
        context_scaler = joblib.load(os.path.join(model_path, 'context_scaler.pkl'))
        y_scaler = joblib.load(os.path.join(model_path, 'y_scaler.pkl'))
    except Exception as e:
        st.error(f"❌ Błąd podczas ładowania enkoderów lub scalerów: {e}")
        st.stop()

    try:
        # Obliczenie rozmiarów wejściowych na podstawie enkoderów i scalerów
        input_size = scaler.scale_.shape[0] + encoder.categories_[0].shape[0]
        context_size = context_scaler.scale_.shape[0] + encoder_constant.categories_[0].shape[0]
        model = TransformerModel(input_size, context_size, d_model=64, nhead=4, num_encoder_layers=4, dim_feedforward=256)
        model.load_state_dict(torch.load(os.path.join(model_path, 'transformer_model.pth'), map_location='cpu'))
        model.eval()
    except Exception as e:
        st.error(f"❌ Błąd podczas ładowania modelu: {e}")
        st.stop()

    return model, encoder, scaler, encoder_constant, context_scaler, y_scaler

def load_data(file_path):
    """
    Ładuje dane z pliku CSV i konwertuje kolumny dat na format datetime.

    Args:
        file_path (str): Ścieżka do pliku CSV.

    Returns:
        pd.DataFrame: Załadowany DataFrame z konwertowanymi kolumnami dat.
    """
    df = pd.read_csv(file_path)
    # Zakładam, że kolumny 6,8,9,10 są niepotrzebne; dostosuj indeksy w razie potrzeby
    df.drop(df.columns[[6, 8, 9, 10]], axis=1, inplace=True)
    df['Data_pomiaru'] = pd.to_datetime(df['Data_pomiaru'], errors='coerce')
    df['Data_ostatniej_awarii'] = pd.to_datetime(df['Data_ostatniej_awarii'], errors='coerce')
    df['Data_prawdziwej_awarii'] = pd.to_datetime(df['Data_prawdziwej_awarii'], errors='coerce')
    return df

def preprocess_data(df) -> tuple:
    """
    Przetwarza dane przez dodanie kolumn liczbowych reprezentujących liczbę dni od daty referencyjnej.

    Args:
        df (pd.DataFrame): DataFrame zawierający surowe dane.

    Returns:
        tuple: (Przetworzony DataFrame, data referencyjna użyta do obliczeń)
    """
    # Oblicz minimalną datę z Data_pomiaru i Data_ostatniej_awarii, ignorując NaT
    min_pomiaru = df['Data_pomiaru'].min()
    min_awarii = df['Data_ostatniej_awarii'].min()
    
    if pd.isnull(min_awarii):
        reference_date = min_pomiaru - pd.Timedelta(days=1)
    else:
        reference_date = min(min_pomiaru, min_awarii) - pd.Timedelta(days=1)
    
    # Dodanie kolumn liczbowych
    df['days_since_measurement'] = (df['Data_pomiaru'] - reference_date).dt.days
    df['days_since_last_failure'] = (df['Data_ostatniej_awarii'] - reference_date).dt.days

    # Obsługa ujemnych wartości w days_since_last_failure
    df['days_since_last_failure'] = df['days_since_last_failure'].apply(lambda x: x if x >= 0 else -1)

    # Poprawne obliczanie days_until_true_failure
    df['days_until_true_failure'] = (df['Data_prawdziwej_awarii'] - reference_date).dt.days
    df['days_until_true_failure'].fillna(-1, inplace=True)
    df.loc[df['days_until_true_failure'] < 0, 'days_until_true_failure'] = -1
    return df, reference_date

def encode_features(df, categorical_columns, numerical_columns):
    """
    Enkoduje cechy kategoryczne za pomocą OneHotEncoder i skaluje cechy numeryczne za pomocą StandardScaler.

    Args:
        df (pd.DataFrame): DataFrame zawierający dane do enkodowania.
        categorical_columns (list): Lista nazw kolumn kategorycznych.
        numerical_columns (list): Lista nazw kolumn numerycznych.

    Returns:
        tuple: (Zakodowane i skalowane cechy, OneHotEncoder obiekt, StandardScaler obiekt)
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categories = encoder.fit_transform(df[categorical_columns])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numerical_columns])
    features = np.concatenate([scaled_features, encoded_categories], axis=1)
    return features, encoder, scaler

def extract_constant_features(df, constant_columns):
    """
    Ekstrahuje stałe cechy dla każdego segmentu rury.

    Args:
        df (pd.DataFrame): DataFrame zawierający dane.
        constant_columns (list): Lista nazw kolumn, które są stałe dla każdego segmentu.

    Returns:
        pd.DataFrame: DataFrame zawierający wyekstrahowane stałe cechy.
    """
    return df.groupby('ID_odcinka', as_index=False)[constant_columns].first()

def encode_constant_features(context_features, categorical_columns, numerical_columns):
    """
    Enkoduje i skaluje stałe cechy kontekstowe.

    Args:
        context_features (pd.DataFrame): DataFrame z cechami kontekstowymi.
        categorical_columns (list): Lista kolumn kategorycznych do enkodowania.
        numerical_columns (list): Lista kolumn numerycznych do skalowania.

    Returns:
        tuple: (Zakodowane i skalowane cechy kontekstowe, OneHotEncoder obiekt, StandardScaler obiekt)
    """
    context_features_for_encoding = context_features.drop(columns=['ID_odcinka'])
    encoder_constant = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categories = encoder_constant.fit_transform(context_features_for_encoding[categorical_columns])
    context_scaler = StandardScaler()
    scaled_numerical = context_scaler.fit_transform(context_features_for_encoding[numerical_columns])
    context_vectors = np.concatenate([scaled_numerical, encoded_categories], axis=1)
    return context_vectors, encoder_constant, context_scaler

def create_sequences(df, features, context_vectors, context_features, reference_date):
    """
    Tworzy sekwencje danych wejściowych dla modelu na podstawie identyfikatorów segmentów rury.

    Args:
        df (pd.DataFrame): DataFrame z danymi.
        features (np.ndarray): Zakodowane i skalowane cechy wejściowe.
        context_vectors (np.ndarray): Zakodowane i skalowane cechy kontekstowe.
        context_features (pd.DataFrame): DataFrame zawierający stałe cechy kontekstowe.
        reference_date (pd.Timestamp): Data referencyjna użyta do obliczeń dat.

    Returns:
        tuple: (Listy sekwencji wejściowych, sekwencji kontekstowych, wartości docelowych)
    """
    X_seq, y_seq, context_seq = [], [], []
    for segment_id in df['ID_odcinka'].unique():
        segment_data = df[df['ID_odcinka'] == segment_id].sort_values('Data_pomiaru')
        sequence_length = 365
        if len(segment_data) < sequence_length:
            continue
        segment_data = segment_data.iloc[:sequence_length]
        segment_indices = segment_data.index
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

def main():
    st.set_page_config(page_title="🔧 Wizualizacja Predykcji Awarii Wodociągów", layout="wide")
    st.title("🔧 Wizualizacja Predykcji Awarii Wodociągów")
    st.markdown("""
    Ta aplikacja wykorzystuje wytrenowany model Transformer do analizy i wizualizacji predykcji awarii rur wodociągowych.
    """)

    # Ścieżki do modeli i danych
    model_path = 'PipeFailurePredict--WaterPrime/Predictions/TransformerModel/models'
    data_path = 'D:/Projects/Aiut/PipeFailurePredict--WaterPrime/Predictions/data/data_failure.csv'

    # Ładowanie modelu i obiektów przetwarzania danych
    model, encoder, scaler, encoder_constant, context_scaler, y_scaler = load_model_and_preprocessors(model_path)

    # Ładowanie danych
    df = load_data(data_path)
    df, reference_date = preprocess_data(df)

    # Definicja kolumn
    categorical_columns = ['Materiał']
    numerical_columns = ['Długość_odcinka', 'Wiek_rury', 'Średnica', 'Przepływ', 'days_since_measurement', 'days_since_last_failure']

    # Enkodowanie i skalowanie cech
    try:
        encoded_categories = encoder.transform(df[categorical_columns])
        scaled_features = scaler.transform(df[numerical_columns])
        features = np.concatenate([scaled_features, encoded_categories], axis=1)
    except Exception as e:
        st.error(f"❌ Błąd podczas enkodowania lub skalowania cech: {e}")
        st.stop()

    # Ekstrakcja i enkodowanie cech kontekstowych
    constant_columns = ['ID_odcinka'] + categorical_columns + ['Średnica', 'Wiek_rury', 'Długość_odcinka']
    context_features = extract_constant_features(df, constant_columns)
    try:
        context_vectors, encoder_constant, context_scaler = encode_constant_features(
            context_features, categorical_columns, ['Średnica', 'Wiek_rury', 'Długość_odcinka']
        )
    except Exception as e:
        st.error(f"❌ Błąd podczas enkodowania cech kontekstowych: {e}")
        st.stop()

    # Tworzenie sekwencji
    try:
        X_seq, context_seq, y_seq = create_sequences(df, features, context_vectors, context_features, reference_date)
    except Exception as e:
        st.error(f"❌ Błąd podczas tworzenia sekwencji: {e}")
        st.stop()

    if not X_seq:
        st.error("❌ Brak wystarczających sekwencji danych do wizualizacji.")
        st.stop()

    X_seq = np.array(X_seq)
    context_seq = np.array(context_seq)
    y_seq = np.array(y_seq)

    # Skalowanie zmiennej celu
    try:
        y_seq_scaled = y_scaler.transform(y_seq.reshape(-1, 1)).flatten()
    except Exception as e:
        st.error(f"❌ Błąd podczas skalowania zmiennej celu: {e}")
        st.stop()

    # Podział na zbiór treningowy i testowy (opcjonalnie)
    try:
        X_train_seq, X_test_seq, y_train_seq, y_test_seq, context_train_seq, context_test_seq = train_test_split(
            X_seq, y_seq_scaled, context_seq, test_size=0.2, random_state=42
        )
    except Exception as e:
        st.error(f"❌ Błąd podczas podziału danych: {e}")
        st.stop()

    # Przygotowanie danych do predykcji
    try:
        X_test_seq_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
        context_test_seq_tensor = torch.tensor(context_test_seq, dtype=torch.float32)
        y_test_seq_tensor = torch.tensor(y_test_seq, dtype=torch.float32)
    except Exception as e:
        st.error(f"❌ Błąd podczas konwersji danych do tensorów: {e}")
        st.stop()

    # Predykcje modelu
    try:
        model.eval()
        with torch.no_grad():
            predictions_scaled = model(X_test_seq_tensor, context_test_seq_tensor).cpu().numpy().flatten()
        predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        actual = y_scaler.inverse_transform(y_test_seq_tensor.cpu().numpy().reshape(-1, 1)).flatten()
    except Exception as e:
        st.error(f"❌ Błąd podczas predykcji: {e}")
        st.stop()

    # Konwersja dni na daty
    try:
        predicted_dates = [reference_date + pd.to_timedelta(day, unit='D') for day in predictions]
        actual_dates = [reference_date + pd.to_timedelta(day, unit='D') for day in actual]
    except Exception as e:
        st.error(f"❌ Błąd podczas konwersji dni na daty: {e}")
        st.stop()

    # Usunięcie czasu z dat
    predicted_dates = [date.date() for date in predicted_dates]
    actual_dates = [date.date() for date in actual_dates]

    # Sprawdzenie długości
    if len(predicted_dates) != len(actual_dates):
        st.error(f"❌ Niezgodność długości: Przewidywane daty: {len(predicted_dates)}, Rzeczywiste daty: {len(actual_dates)}")
        st.stop()

    # Tworzenie DataFrame do wizualizacji
    visualization_data = pd.DataFrame({
        'Przewidywana Data Awarii': predicted_dates,
        'Rzeczywista Data Awarii': actual_dates
    })

    # Obliczenie błędów
    errors = (predictions - actual)
    errors_years = errors / 365.25

    # Obliczenie RMSE
    try:
        mse = mean_squared_error(actual, predictions)
        rmse_years = np.sqrt(mse) / 365.25
    except Exception as e:
        st.error(f"❌ Błąd podczas obliczania RMSE: {e}")
        st.stop()

    # Wczytanie oryginalnych cech
    try:
        original_features = ['Materiał', 'Wiek_rury', 'Średnica', 'Długość_odcinka', 'Przepływ']
        # Upewnij się, że liczba próbek nie przekracza dostępnych danych
        df_features = df[df['days_until_true_failure'] != -1][original_features].iloc[:len(predictions)].reset_index(drop=True)
        df_features['Error'] = errors_years
    except Exception as e:
        st.error(f"❌ Błąd podczas przygotowywania oryginalnych cech: {e}")
        st.stop()

    # Wizualizacje
    st.header("📈 Porównanie Przewidywanych i Rzeczywistych Dat Awarii")
    st.dataframe(visualization_data.head(20))

    fig1, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(visualization_data['Przewidywana Data Awarii'].head(40), label='Przewidywane Daty Awarii', marker='o', linestyle='--', color='blue')
    ax1.plot(visualization_data['Rzeczywista Data Awarii'].head(40), label='Rzeczywiste Daty Awarii', marker='x', linestyle='-', color='red')
    ax1.set_xlabel('Próbki', fontsize=12)
    ax1.set_ylabel('Daty Awarii', fontsize=12)
    ax1.set_title('Porównanie Przewidywanych i Rzeczywistych Dat Awarii', fontsize=14)
    ax1.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)

    st.header("📊 Wydajność Modelu")
    st.metric("📉 Średni Błąd Kwadratowy (RMSE)", f"{rmse_years:.2f} lat")

    st.header("🔍 Rozkład Błędów Predykcji")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.histplot(errors_years, bins=30, kde=True, ax=ax2, color='purple')
    ax2.set_xlabel('Błąd (lata)', fontsize=12)
    ax2.set_ylabel('Liczba Prób', fontsize=12)
    ax2.set_title('Rozkład Błędów Predykcji', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig2)

    st.header("🔎 Wpływ Cech na Predykcję")
    for feature in original_features:
        fig, ax = plt.subplots(figsize=(10, 6))
        if df_features[feature].dtype == 'object':
            mean_errors = df_features.groupby(feature)['Error'].mean().reset_index()
            sns.barplot(x=feature, y='Error', data=mean_errors, ax=ax, palette='viridis')
            ax.set_xlabel(feature, fontsize=12)
        else:
            sns.scatterplot(x=df_features[feature], y=df_features['Error'], ax=ax, palette='viridis')
            ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Błąd Predykcji (lata)', fontsize=12)
        ax.set_title(f'Wpływ {feature} na Błąd Predykcji', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)

    st.sidebar.header("🔄 Informacje o Modelu")
    st.sidebar.write(f"**Średni Błąd Kwadratowy (RMSE):** {rmse_years:.2f} lat")
    st.sidebar.write(f"**Liczba Prób Testowych:** {len(predictions)}")

if __name__ == "__main__":
    main()
