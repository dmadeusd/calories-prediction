# Calories Burnt Predictor

## Project Overview
This project is an end-to-end Machine Learning web application designed to estimate the number of calories a person burns during physical exercise. 

The system leverages personal physiological data and exercise metrics to make accurate predictions. Alongside the core machine learning prediction, the application also computes the user's **Basal Metabolic Rate (BMR)** using the Mifflin-St Jeor equation.

The final web application is built using **Streamlit** with a custom-designed, modern glassmorphism user interface.


## Dataset Information
The model was trained on the **Calories Burnt Prediction** dataset.

- **Source:** [Kaggle - Calories Burnt Prediction by Ruchika Kumbhar](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction)
- **Description:** The dataset contains the physiological and exercise data of various individuals, along with the actual calories burned.

### Key Features Used:
* **Gender**: Male / Female
* **Age**: Age of the individual in years
* **Height**: Height in centimeters (cm)
* **Weight**: Weight in kilograms (kg)
* **Duration**: Total duration of the workout in minutes
* **Heart_Rate**: Average heart rate during the workout (beats per minute - bpm)
* **Body_Temp**: Body temperature during the workout (Celsius)
* **Calories**: Total calories burned (Target Variable)

## Machine Learning Models Evaluated
During the development phase, several machine learning regression models were trained and evaluated to find the best fit for predicting calories burned. The experiments are documented in the Jupyter Notebooks included in the repository:

1. **Decision Tree Regressor** (`DTree-Regressor.ipynb`)
2. **Support Vector Regressor** (`SVR.ipynb`)
3. **Random Forest Regressor** (`RF-Regressor.ipynb`) - **Selected Model**

The **Random Forest Regressor** was chosen as the final model due to its high accuracy, robustness against overfitting, and ability to handle non-linear relationships effectively. The trained model is serialized and stored as `rf_model.sav` in the `Model/` directory.

## Web Application (Streamlit)
The web application provides an intuitive and modern interface for users to input their data and get real-time predictions. 

### Features:
- Live BMR Calculation: As you input your personal information (Gender, Age, Height, Weight), your daily BMR is automatically calculated and displayed in real-time.
- Calories Prediction: After entering your workout details (Duration, Avg Heart Rate), click the predict button to estimate how many calories you burned.