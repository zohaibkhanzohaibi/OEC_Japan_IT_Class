import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess data
def load_data():
    # Load energy data
    energy_url = "../datasets/owid-energy-data.csv"
    energy_df = pd.read_csv(energy_url)
    
    # Filter for a specific country and relevant columns
    energy_df = energy_df[energy_df['country'] == 'United States']
    energy_df = energy_df[['year', 'primary_energy_consumption']]
    
    # Simulate temperature data
    np.random.seed(42)
    energy_df['temperature'] = np.random.normal(20, 5, size=len(energy_df))
    
    # Create time-based features
    energy_df['decade'] = (energy_df['year'] // 10) * 10
    energy_df.dropna(inplace=True)
    
    return energy_df

# Main app
def main():
    st.title("Energy Consumption Analysis Dashboard")
    
    # Load data
    df = load_data()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["EDA", "Modeling", "Prediction"])
    
    # -------------------------
    # EDA Tab
    # -------------------------
    with tab1:
        st.header("Exploratory Data Analysis")
        
        # Summary statistics
        st.subheader("Data Summary")
        st.write(df.describe())
        
        # Time series plot
        st.subheader("Energy Consumption Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df, x='year', y='primary_energy_consumption', ax=ax)
        ax.set_ylabel("Energy Consumption (TWh)")
        st.pyplot(fig)
        
        # Temperature distribution
        st.subheader("Temperature Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df['temperature'], kde=True, ax=ax)
        st.pyplot(fig)
        
        # Correlation matrix
        st.subheader("Feature Correlation Matrix")
        corr_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    # -------------------------
    # Modeling Tab
    # -------------------------
    with tab2:
        st.header("Energy Consumption Prediction Model")
        
        # Prepare data
        X = df[['year', 'decade', 'temperature']]
        y = df['primary_energy_consumption']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = SGDRegressor(
            max_iter=1000, 
            tol=1e-3,
            learning_rate='adaptive',
            eta0=0.01,
            penalty='l2'
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Show metrics
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{mse:.2f}")
        col2.metric("R-squared Score", f"{r2:.2f}")
        
        # Actual vs Predicted plot
        st.subheader("Actual vs Predicted Values")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        st.pyplot(fig)
        
        # Show coefficients
        st.subheader("Model Coefficients")
        coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Weight': model.coef_
        })
        st.dataframe(coefficients)
    
    # -------------------------
    # Prediction Tab
    # -------------------------
    with tab3:
        st.header("Energy Consumption Predictor")
        
        # Create input widgets
        year = st.slider("Select Year", 
                        int(df['year'].min()), 
                        int(df['year'].max()),
                        int(df['year'].median()))
        
        temp = st.number_input("Enter Temperature (°C)", 
                             min_value=float(df['temperature'].min()),
                             max_value=float(df['temperature'].max()),
                             value=20.0)
        
        # Calculate decade
        decade = (year // 10) * 10
        
        # Create prediction input
        input_data = pd.DataFrame([[year, decade, temp]],
                                columns=['year', 'decade', 'temperature'])
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        if st.button("Predict Energy Consumption"):
            prediction = model.predict(input_scaled)[0]
            st.success(f"Predicted Energy Consumption: {prediction:.2f} TWh")
            
            # Show historical context
            st.subheader("Historical Context")
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x='year', y='primary_energy_consumption')
            ax.axvline(year, color='red', linestyle='--', label='Prediction Year')
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()