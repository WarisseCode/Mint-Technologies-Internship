# King County House Price Prediction

## Problem Statement
This project addresses the real estate valuation challenge by building a predictive machine learning system. The goal is to accurately predict residential house prices in King County based on property characteristics, empowering buyers, sellers, and real estate professionals with data-driven pricing insights.

## Dataset
- **Source:** King County House Sales dataset (historical housing data containing 21,613 records).
- **Features:** Includes quantitative property metrics (such as square footage, number of bedrooms/bathrooms, floors, grade, year built, and geographical coordinates).

## Approach
The project follows a rigorous 5-milestone Data Science lifecycle:
1. **Exploration & Cleaning (`01_exploration_cleaning.ipynb`):** Conducted Exploratory Data Analysis (EDA), inspected missing values, cleaned raw records, and generated core visualizations (target distribution, correlation heatmap, scatter plots).
2. **Feature Engineering (`02_feature_engineering.ipynb`):** Encoded categorical features (such as `zipcode`), scaled numerical variables using `StandardScaler`, and derived domain-specific features like `house_age`.
3. **Baseline Model (`03_baseline_model.ipynb`):** Established a Linear Regression baseline model, splitting data into training and testing sets, and evaluated performance using MSE, RMSE, and $R^2$ scores.
4. **Model Comparison (`04_advanced_models.ipynb`):** Trained advanced ensemble algorithms (Random Forest and Gradient Boosting) and optimized hyperparameters using `GridSearchCV`.
5. **Capstone Deployment (`app.py`):** Serialized the best-performing model using `joblib` and built an interactive web application with Streamlit.

## Results
- **Baseline Model (Linear Regression):** Achieved an $R^2$ score of **0.8973** and an RMSE of **$124,617.00**, demonstrating strong initial predictive power.
- **Advanced Models:** Ensemble models (Random Forest and Gradient Boosting) captured non-linear patterns and further refined predictive accuracy through hyperparameter tuning.

## How to Run
To run this application locally on your machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd Mint_Internship

2. Install the required dependencies:
```bash
pip install -r requirements.txt


*(If you encounter an error with command execution paths, you can also install packages individually using `python -m pip install streamlit pandas numpy scikit-learn joblib`)*
3. Launch the Streamlit web application:
```bash
python -m streamlit run app.py

