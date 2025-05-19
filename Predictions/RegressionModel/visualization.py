import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

class Visualization:
    """
    Class containing methods for visualizing model predictions and feature impacts.
    """

    @staticmethod
    def visualize_predictions(visualization_data):
        """
        Visualizes the comparison between predicted and actual failure dates.

        Args:
            visualization_data (DataFrame): DataFrame containing predicted and actual failure dates.
        """
        st.subheader("📈 Porównanie Przewidywanych i Rzeczywistych Dat Awarii")
        st.dataframe(visualization_data.head(20))

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(visualization_data['Sample'][:40], visualization_data['Predicted Failure Date'][:40],
                label='Przewidywane Daty Awarii', marker='o', linestyle='--', color='blue')
        ax.plot(visualization_data['Sample'][:40], visualization_data['Actual Failure Date'][:40],
                label='Rzeczywiste Daty Awarii', marker='x', linestyle='-', color='red')
        ax.set_xlabel('Próbki', fontsize=12)
        ax.set_ylabel('Daty Awarii', fontsize=12)
        ax.set_title('Porównanie Przewidywanych i Rzeczywistych Dat Awarii', fontsize=14)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    @staticmethod
    def visualize_errors(y_true, y_pred):
        """
        Visualizes the distribution of prediction errors in years.

        Args:
            y_true (array): Actual target values.
            y_pred (array): Predicted target values.
        """
        errors = y_pred - y_true
        errors_years = errors / 365.25
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(errors_years, bins=30, kde=True, ax=ax, color='purple')
        ax.set_xlabel('Błąd (lata)', fontsize=12)
        ax.set_ylabel('Liczba Prób', fontsize=12)
        ax.set_title('Rozkład Błędów Predykcji', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)

    @staticmethod
    def display_model_performance(mse, y_true):
        """
        Displays model performance metrics in terms of RMSE in years.

        Args:
            mse (float): Mean Squared Error.
            y_true (array): Actual target values.
        """
        st.subheader("📊 Wydajność Modelu")
        rmse_years = np.sqrt(mse) / 365.25
        st.metric("📉 Średni Błąd Kwadratowy (RMSE)", f"{rmse_years:.2f} lat")

    @staticmethod
    def visualize_feature_impact(df_features, errors, reference_date):
        """
        Visualizes the impact of features on prediction errors.

        Args:
            df_features (DataFrame): DataFrame with original features.
            errors (array): Prediction errors.
            reference_date (Timestamp): Reference date used in preprocessing.
        """
        st.subheader("🔎 Wpływ Cech na Predykcję")

        df_features['Error'] = errors / 365.25
        original_features = ['Materiał', 'Wiek_rury', 'Średnica', 'Długość_odcinka', 'Przepływ']

        for feature in original_features:
            fig, ax = plt.subplots(figsize=(10, 6))
            if df_features[feature].dtype == 'object':
                mean_errors = df_features.groupby(feature)['Error'].mean().reset_index()
                sns.barplot(x=feature, y='Error', data=mean_errors, ax=ax)
                ax.set_xlabel(feature, fontsize=12)
            else:
                sns.scatterplot(x=df_features[feature], y=df_features['Error'], ax=ax)
                ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Błąd Predykcji (lata)', fontsize=12)
            ax.set_title(f'Wpływ {feature} na Błąd Predykcji', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
