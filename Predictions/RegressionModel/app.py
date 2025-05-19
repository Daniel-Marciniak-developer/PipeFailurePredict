import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import joblib
import os
from visualization import Visualization

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

def load_model(path='D:\Projects\Aiut\PipeFailurePredict--WaterPrime\Predictions\RegressionModel\models'):
    """
    Loads the trained model and preprocessing objects.

    Args:
        path (str): Path to the model and preprocessing objects.

    Returns:
        model (nn.Module): Loaded model.
        scaler_X (StandardScaler): Scaler for features.
        encoder_X (OneHotEncoder): Encoder for features.
        scaler_y (StandardScaler): Scaler for target variable.
        encoder_constant (OneHotEncoder): Encoder for context features.
        scaler_constant (StandardScaler): Scaler for context features.
        feature_names (list): Names of features.
        context_feature_names (list): Names of context features.
        device (torch.device): Device to run the model on.
    """
    try:
        feature_names = joblib.load(os.path.join(path, 'feature_names.pkl'))
        context_feature_names = joblib.load(os.path.join(path, 'context_feature_names.pkl'))
        scaler_X = joblib.load(os.path.join(path, 'scaler_X.pkl'))
        encoder_X = joblib.load(os.path.join(path, 'encoder_X.pkl'))
        scaler_y = joblib.load(os.path.join(path, 'scaler_y.pkl'))
        encoder_constant = joblib.load(os.path.join(path, 'encoder_constant.pkl'))
        scaler_constant = joblib.load(os.path.join(path, 'scaler_constant.pkl'))
    except Exception as e:
        st.error(f"❌ Wystąpił błąd podczas ładowania scalera lub enkodera: {e}")
        raise e

    try:
        model = RegressionModel(input_size=len(feature_names) + len(context_feature_names))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_state = torch.load(os.path.join(path, 'regression_model.pth'), map_location=device, weights_only=True)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
    except Exception as e:
        st.error(f"❌ Wystąpił błąd podczas ładowania modelu: {e}")
        raise e

    return model, scaler_X, encoder_X, scaler_y, encoder_constant, scaler_constant, feature_names, context_feature_names, device

def main():
    st.set_page_config(page_title="🔧 Predykcja Awarii Wodociągów")
    st.title("🔧 Predykcja Awarii Wodociągów")
    st.markdown("""
    Aplikacja wykorzystuje model sieci neuronowej do przewidywania dat awarii wodociągów na podstawie danych historycznych.
    Model jest zoptymalizowany tak, aby przewidywać awarie wcześniej, co pozwala na proaktywne działania konserwacyjne.
    """, unsafe_allow_html=True)

    st.sidebar.header("🔄 Ładowanie Danych")
    uploaded_file = st.sidebar.file_uploader("Wybierz plik CSV z danymi", type=["csv"])

    if uploaded_file is not None:
        with st.spinner('Ładowanie danych...'):
            df = pd.read_csv(uploaded_file)
            st.subheader("📊 Początkowe Dane")
            st.dataframe(df.head())

        required_columns = ['ID_odcinka', 'Data_pomiaru', 'Materiał', 'Wiek_rury', 'Średnica', 'Długość_odcinka',
                            'Przepływ', 'Data_ostatniej_awarii', 'Data_prawdziwej_awarii']
        if not all(col in df.columns for col in required_columns):
            st.error("❌ Plik CSV nie zawiera wymaganych kolumn.")
            return

        df.dropna(subset=required_columns, inplace=True)
        df['Data_pomiaru'] = pd.to_datetime(df['Data_pomiaru'], errors='coerce')
        df['Data_ostatniej_awarii'] = pd.to_datetime(df['Data_ostatniej_awarii'], errors='coerce')
        df['Data_prawdziwej_awarii'] = pd.to_datetime(df['Data_prawdziwej_awarii'], errors='coerce')
        df.dropna(subset=['Data_pomiaru', 'Data_ostatniej_awarii', 'Data_prawdziwej_awarii'], inplace=True)

        reference_date = min(df['Data_pomiaru'].min(), df['Data_ostatniej_awarii'].min()) - pd.Timedelta(days=1)
        df['days_since_measurement'] = (df['Data_pomiaru'] - reference_date).dt.days
        df['days_since_last_failure'] = (df['Data_ostatniej_awarii'] - reference_date).dt.days
        df['days_until_true_failure'] = (df['Data_prawdziwej_awarii'] - df['Data_pomiaru']).dt.days
        df['days_until_true_failure'].fillna(-1, inplace=True)
        df.loc[df['days_until_true_failure'] < 0, 'days_until_true_failure'] = -1
        df['age_diameter_ratio'] = df['Wiek_rury'] / (df['Średnica'] + 1)
        df_features = df[df['days_until_true_failure'] != -1]

        if df_features.empty:
            st.error("❌ Brak danych z prawdziwymi awariami w przesłanym pliku.")
            return

        try:
            model, scaler_X, encoder_X, scaler_y, encoder_constant, scaler_constant, feature_names, context_feature_names, device = load_model()
        except Exception as e:
            st.error(f"❌ Wystąpił błąd podczas ładowania modelu: {e}")
            return

        categorical_columns = ['Materiał']
        numerical_columns = ['Długość_odcinka', 'Wiek_rury', 'Średnica', 'Przepływ',
                             'days_since_last_failure', 'age_diameter_ratio']

        try:
            df_features_categorical = df_features[categorical_columns].copy()
            df_features_categorical.columns = encoder_X.feature_names_in_
            encoded_categories = encoder_X.transform(df_features_categorical)
            scaled_features = scaler_X.transform(df_features[numerical_columns])
            X_sequential = np.concatenate([scaled_features, encoded_categories], axis=1)
        except Exception as e:
            st.error(f"❌ Wystąpił błąd podczas kodowania cech: {e}")
            return

        try:
            constant_columns = ['ID_odcinka'] + categorical_columns + ['Średnica', 'Wiek_rury', 'Długość_odcinka']
            context_features = df_features[constant_columns].drop_duplicates(subset=['ID_odcinka'])
            context_features_numerical = context_features[['Średnica', 'Wiek_rury', 'Długość_odcinka']]
            context_features_categorical = context_features[categorical_columns]
            context_features_categorical.columns = encoder_constant.feature_names_in_
            scaled_numerical_constant = scaler_constant.transform(context_features_numerical)
            encoded_categories_constant = encoder_constant.transform(context_features_categorical)
            context_vectors = np.concatenate([scaled_numerical_constant, encoded_categories_constant], axis=1)
            context_dict = dict(zip(context_features['ID_odcinka'], context_vectors))
            context_X = np.array([context_dict[id_] for id_ in df_features['ID_odcinka']])
        except Exception as e:
            st.error(f"❌ Wystąpił błąd podczas kodowania cech kontekstowych: {e}")
            return

        X_final = np.hstack([X_sequential, context_X])
        y = df_features['days_until_true_failure'].values

        try:
            y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
        except Exception as e:
            st.error(f"❌ Wystąpił błąd podczas skalowania celu: {e}")
            return

        X_tensor = torch.tensor(X_final, dtype=torch.float32).to(device)

        try:
            with torch.no_grad():
                predictions_scaled = model(X_tensor).cpu().numpy().flatten()
        except Exception as e:
            st.error(f"❌ Wystąpił błąd podczas predykcji: {e}")
            return

        try:
            predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            actual = y
        except Exception as e:
            st.error(f"❌ Wystąpił błąd podczas odwracania skalowania: {e}")
            return

        predictions = np.where(predictions < 0, 0, predictions)

        try:
            predicted_dates = [(df_features['Data_pomiaru'].iloc[i] + pd.to_timedelta(predictions[i], unit='D')).date()
                               for i in range(len(predictions))]
            actual_dates = [date.date() for date in df_features['Data_prawdziwej_awarii']]
        except Exception as e:
            st.error(f"❌ Wystąpił błąd podczas obliczania dat predykcji: {e}")
            return

        visualization_data = pd.DataFrame({
            'Sample': range(1, len(predictions) + 1),
            'Predicted Failure Date': predicted_dates,
            'Actual Failure Date': actual_dates
        })

        try:
            mse = mean_squared_error(actual, predictions)
            Visualization.display_model_performance(mse, actual)
        except Exception as e:
            st.error(f"❌ Wystąpił błąd podczas oceny modelu: {e}")
            return

        try:
            Visualization.visualize_predictions(visualization_data)
            Visualization.visualize_errors(actual, predictions)
            errors = predictions - actual
            df_original_features = df_features[['Materiał', 'Wiek_rury', 'Średnica',
                                                'Długość_odcinka', 'Przepływ']].copy()
            Visualization.visualize_feature_impact(df_original_features, errors, reference_date)
        except Exception as e:
            st.error(f"❌ Wystąpił błąd podczas wizualizacji wyników: {e}")
            return

        st.sidebar.header("🔮 Prognozowanie Nowych Danych")
        st.sidebar.markdown("Wprowadź wartości cech, aby uzyskać prognozę daty awarii.")

        new_data = {}
        for col in ['Długość_odcinka', 'Wiek_rury', 'Średnica', 'Przepływ']:
            new_data[col] = st.sidebar.number_input(f"📏 {col}", value=float(df[col].mean()), step=1.0)

        new_data['Materiał'] = st.sidebar.selectbox("🔧 Materiał", options=df['Materiał'].unique())

        new_last_failure_date = st.sidebar.date_input("🛠️ Data Ostatniej Awarii",
                                                      value=pd.to_datetime("today").normalize(),
                                                      min_value=pd.to_datetime('1900-01-01'),
                                                      max_value=pd.to_datetime("today").normalize())
        
        new_measurement_date = pd.to_datetime("today").normalize()
        st.sidebar.write(f"📅 **Data Pomiaru (dzisiaj):** {new_measurement_date.date()}")

        if st.sidebar.button("📈 Przewiduj Awarię"):
            try:
                days_since_last_failure = (pd.to_datetime(new_last_failure_date) - reference_date).days

                new_features_values = [
                    new_data['Długość_odcinka'],
                    new_data['Wiek_rury'],
                    new_data['Średnica'],
                    new_data['Przepływ'],
                    days_since_last_failure,
                    new_data['Wiek_rury'] / (new_data['Średnica'] + 1)
                ]

                new_encoded = encoder_X.transform(pd.DataFrame({'Materiał': [new_data['Materiał']]}))
                new_scaled = scaler_X.transform([new_features_values])
                new_features_sequential = np.concatenate([new_scaled.flatten(), new_encoded.flatten()])

                new_context_numerical = [new_data['Średnica'], new_data['Wiek_rury'], new_data['Długość_odcinka']]
                new_scaled_context_numerical = scaler_constant.transform([new_context_numerical])
                new_encoded_context = encoder_constant.transform([[new_data['Materiał']]])
                new_context = np.concatenate([new_scaled_context_numerical.flatten(), new_encoded_context.flatten()])

                new_X = np.hstack([new_features_sequential, new_context])

                new_X_tensor = torch.tensor(new_X, dtype=torch.float32).to(device)

                with torch.no_grad():
                    new_pred_scaled = model(new_X_tensor).cpu().numpy().flatten()

                new_pred = scaler_y.inverse_transform(new_pred_scaled.reshape(-1, 1)).flatten()[0]

                if new_pred < 0:
                    new_pred = 0

                new_pred_date = (new_measurement_date + pd.to_timedelta(new_pred, unit='D')).date()

                st.sidebar.success(f"✅ **Przewidywana Data Awarii:** {new_pred_date}")
            except Exception as e:
                st.sidebar.error(f"❌ Wystąpił błąd podczas przewidywania awarii: {e}")

if __name__ == "__main__":
    main()
