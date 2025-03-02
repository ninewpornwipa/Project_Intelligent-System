import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# Load Machine Learning Models
logistic_model = joblib.load("logistic_regression_model.pkl")
random_forest_model = joblib.load("random_forest_model.pkl")
svm_model = joblib.load("svm_model.pkl")


# สร้าง Sidebar Menu
st.sidebar.title("📌Menu")
page = st.sidebar.radio(
    "🔍 Select menu",
    [
        "📖 Machine Learning",
        "📊 Water Potability ",
        "🤖 Neural Network",
        "🌡️ Temperature Forecast"
    ]
)






st.title("Machine Learning & Neural Network Web Application📊")



#page 1
if page == "📖 Machine Learning":
    st.header("Machine Learning Overview")
    
    st.write("""
    
## 1️⃣ Data Preparation

### 🔹 Loading Data
- Used `pandas` to read data from the file `water_potability.csv`.
- Load dataset from: [Kaggle](https://www.kaggle.com/).
- Inspected the dataset to understand the structure and types of data.

### 🔹 Handling Missing Values
- Used `fillna()` method to replace missing values with the mean of each column.
- Ensures data completeness and reduces bias caused by missing values.
- Checked for outliers using **Boxplot** and handled extreme values.

### 🔹 Separating Features and Labels
- **X (Features):** pH, Hardness, Solids, Chloramines, Conductivity, etc.
- **y (Label):** Potability (0 = Not Drinkable, 1 = Drinkable).
- Checked feature correlation to remove redundant variables.

### 🔹 Train-Test Split
- Used `train_test_split()` to split the dataset (80% training, 20% testing).
- Applied `stratify=y` to maintain class distribution.
- Implemented `StratifiedKFold()` for balanced data splitting.

### 🔹 Feature Scaling
- Used `StandardScaler()` to normalize feature values.
- Applied `fit_transform()` on training data and `transform()` on test data.
- Helps improve model performance by standardizing feature ranges.

---

## 2️⃣ Theory of Algorithms Used

### 🔹 1. Random Forest Classifier 🌲
- Uses an ensemble of **Decision Trees** for robust classification.
- Reduces overfitting and performs well on datasets with multiple important features.
- **Weakness:** Can be slow with large datasets.

### 🔹 2. Logistic Regression 📊
- Uses a **sigmoid function** for binary classification.
- Helps in understanding **feature importance**.
- **Weakness:** Performs poorly on non-linearly separable data.

### 🔹 3. Support Vector Machine (SVM) 📉
- Finds the **optimal hyperplane** for classification.
- Uses **Kernel Trick** to handle **non-linear data**.
- **Weakness:** Computationally expensive for large datasets.

---

## 3️⃣ Model Development Steps

### ✅ Training the Model
- Used `RandomForestClassifier(n_estimators=100, random_state=42)`.
- Trained using `rf_model.fit(X_train, y_train)`.
- Applied **GridSearchCV** to fine-tune hyperparameters.

### ✅ Making Predictions
- Used `predict()` to generate predictions on test data.
- Stored predictions for evaluation.

### ✅ Evaluating the Model
- Used `accuracy_score` and `classification_report`.
- Evaluated **Precision, Recall, and F1-score**.
- Used **Confusion Matrix** for better insights into classification errors.

### ✅ Comparing Model Performance
- Compared **Random Forest, Logistic Regression, and SVM**.
- Analyzed **ROC-AUC Score** and **Precision-Recall Curve**.
- Determined the best model based on real-world generalization ability.

---
 """)
    st.write("""
    
    ### 🔗 References
    - Dataset Source: [Kaggle](https://www.kaggle.com/)
    - Scikit-learn Documentation: [Scikit-learn](https://scikit-learn.org/stable/documentation.html)
    """)
# ----------------------------------------------------------
    
#page 2
if page == "📊 Water Potability ":
    st.subheader("🔍 Water Potability (Machine Learning)")
    df = pd.read_csv("water_potability.csv")
    feature_ranges = {col: (df[col].min(), df[col].max()) for col in df.columns if col != "Potability"}
    model_options = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
        "SVM": svm_model
    }
    features = list(feature_ranges.keys())
    for feature in features:
        if feature not in st.session_state:
            st.session_state[feature] = round(df[feature].mean(), 2)
    if st.button("Random Feature - Unsafe Water"):
        for feature in features:
            min_val, max_val = feature_ranges[feature]
            if random.random() < 0.7:  # 70% probability to generate unsafe water
                st.session_state[feature] = round(random.uniform(min_val, min_val + (max_val - min_val) * 0.3), 2)
            else:
                st.session_state[feature] = round(random.uniform(min_val, max_val), 2)
    input_data = []
    for feature in features:
        value = st.number_input(
            f"{feature}",
            min_value=feature_ranges[feature][0],
            max_value=feature_ranges[feature][1],
            value=st.session_state[feature],
            key=f"feature_{feature}"
        )
        st.session_state[feature] = value
        input_data.append(value)
    if st.button("Prediction "):
        predictions = {}
        results = []
        for model_name, model in model_options.items():
            prediction = model.predict(np.array([input_data]))[0]
            result_text = "✅ Drinkable Water" if prediction == 1 else "❌ I Can't Drink Water."
            predictions[model_name] = result_text
            results.append((model_name, prediction))
        st.success("📊 **Results From All Models**")
        for model_name, result_text in predictions.items():
            st.info(f"🤖 **{model_name}**: {result_text}")
        st.subheader("📈 Comparison Of the Results Of Each Model")
        model_names = [r[0] for r in results]
        prediction_values = [r[1] for r in results]
        fig, ax = plt.subplots()
        ax.bar(model_names, prediction_values, color=['blue', 'green', 'red'])
        ax.set_ylabel("ผลการพยากรณ์ (0 = น้ำดื่มไม่ได้, 1 = น้ำดื่มได้)")
        ax.set_title("ผลการพยากรณ์จากแต่ละโมเดล")
        ax.set_ylim(0, 1.5)
        for i, v in enumerate(prediction_values):
            ax.text(i, v + 0.05, str(v), ha='center', fontsize=12)
        st.pyplot(fig)

#page 3
if page == "🤖 Neural Network":
    st.header("Neural Network Overview")
    st.write("""
    ### 📊 Data Preparation
    - Read dataset `weather_rain_modified.csv` from [Kaggle](https://www.kaggle.com/).
    - Filled missing values using column-wise mean.
    - Applied Min-Max Scaling for normalization.
    - Split dataset into **80% Training** and **20% Testing**.

    ### 📚 Algorithm Theory
    - Implemented **Fully Connected Neural Network** with TensorFlow/Keras.
    - Used **ReLU Activation** in hidden layers, **Linear Activation** in the output layer.
    - Optimized using **Adam Optimizer** and trained with **Backpropagation & Gradient Descent**.

    ### 🛠 Model Development Steps
    ```python
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
    ```

    ### 📈 Model Evaluation
    ```python
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test MAE: {mae}")
    ```

    ### 🎯 Model Prediction
    ```python
    predictions = model.predict(X_test)
    ```

    ✅ **Neural Network is Ready for Temperature Prediction!**
    """)
    st.write("""
    ---
    ### 🔗 References
    - Dataset Source: [Kaggle](https://www.kaggle.com/)
    - TensorFlow/Keras Documentation: [TensorFlow](https://www.tensorflow.org/api_docs)
    """)




if page == "🌡️ Temperature Forecast":
    model_path = r"C:\Users\Asus\Desktop\Project_IS\Temperature_Forecast.pkl"

    # 🔹 โหลดโมเดล
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = None

    # 🔹 โหลด MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit([[0, 0, 0, 900], [100, 50, 100, 1100]])
    scaler_y.fit([[10], [40]])

    st.subheader("🌡️ Temperature Forecast (Neural Network)")

    st.write("Please enter the values to predict the temperature (°C)")

    # 🔹 ช่องกรอกข้อมูล
    humidity = st.number_input("Humidity", min_value=30.0, max_value=100.0, step=1.0)
    wind_speed = st.number_input("Wind Speed", min_value=1.0, max_value=50.0, step=1.0)
    cloud_cover = st.number_input("Cloud Cover", min_value=1.0, max_value=100.0, step=1.0)
    pressure = st.number_input("Pressure", min_value=900.0, max_value=1100.0, step=1.0)


        # ✅ ปุ่มทำนาย
    if st.button("Predict"):
            input_data = np.array([[humidity, wind_speed, cloud_cover, pressure]])
            input_data_scaled = scaler_X.transform(input_data)

            # 🔹 ใช้โมเดลทำการพยากรณ์
            prediction_scaled = model.predict(input_data_scaled)

            # 🔹 ทำ Inverse Transform ให้ค่ากลับมาเป็นองศาเซลเซียส
            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

            # 🔹 แสดงผลลัพธ์
            st.success(f"🌡️ Predicted Temperature: **{prediction[0][0]:.2f} °C**")
        # 🔗 References
            st.write("---")
            st.write("### 🔗 References")
            st.write("- Dataset Source: [Kaggle](https://www.kaggle.com/)")
            st.write("- TensorFlow/Keras Documentation: [TensorFlow](https://www.tensorflow.org/api_docs)")