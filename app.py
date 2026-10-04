import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Canteen AI Predictor",
    page_icon="🍱",
    layout="centered"
)


# --------------------------------------------------
# CUSTOM STYLE
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CREATE DATASET
# --------------------------------------------------

np.random.seed(42)

n = 100

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
food_types = ["Rice_Dal", "Chole_Rice", "Rajma_Rice", "Poha", "Sandwich"]

students_present = np.random.randint(550, 901, n)

previous_sales = np.clip(
    students_present * np.random.uniform(0.17, 0.25, n)
    + np.random.normal(0, 8, n),
    80, 230
).round().astype(int)

day = np.random.choice(days, n)
temperature = np.random.randint(22, 36, n)
exam_day = np.random.choice([0, 1], n, p=[0.85, 0.15])
food_type = np.random.choice(food_types, n)

food_effect = {
    "Rice_Dal": 5,
    "Chole_Rice": 8,
    "Rajma_Rice": 7,
    "Poha": -5,
    "Sandwich": -2
}

day_effect = {
    "Monday": 4,
    "Tuesday": 2,
    "Wednesday": 0,
    "Thursday": 1,
    "Friday": -6
}

demand = (
    students_present * 0.20
    + previous_sales * 0.25
    + np.array([food_effect[x] for x in food_type])
    + np.array([day_effect[x] for x in day])
    - exam_day * 12
    + np.where(temperature > 32, -4, 0)
    + np.random.normal(0, 6, n)
)

meals_sold = np.clip(demand, 80, 220).round().astype(int)

data = pd.DataFrame({
    "students_present": students_present,
    "previous_sales": previous_sales,
    "day": day,
    "temperature": temperature,
    "exam_day": exam_day,
    "food_type": food_type,
    "meals_sold": meals_sold
})


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

X = data.drop("meals_sold", axis=1)
y = data["meals_sold"]


categorical_features = ["day", "food_type"]

numeric_features = [
    "students_present",
    "previous_sales",
    "temperature",
    "exam_day"
]


# --------------------------------------------------
# PREPROCESSOR
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

model.fit(X_train, y_train)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

test_predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    test_predictions
)

r2 = r2_score(
    y_test,
    test_predictions
)


# --------------------------------------------------
# FINAL MODEL
# --------------------------------------------------

final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

final_model.fit(X, y)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🍱 Canteen AI Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-based meal demand prediction for reducing avoidable food waste'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SDG INFORMATION
# --------------------------------------------------

st.info(
    "🌱 SDG 12 — Responsible Consumption and Production\n\n"
    "This project uses machine learning to estimate meal demand "
    "so that school canteens can plan food preparation more efficiently."
)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📊 Enter Canteen Information</div>',
    unsafe_allow_html=True
)

students_present = st.number_input(
    "👨‍🎓 Students Present",
    min_value=1,
    max_value=2000,
    value=820
)

previous_sales = st.number_input(
    "📈 Previous Sales",
    min_value=0,
    max_value=1000,
    value=175
)

day = st.selectbox(
    "📅 Day",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
)

temperature = st.number_input(
    "🌡️ Temperature (°C)",
    min_value=0,
    max_value=50,
    value=30
)

exam_day = st.selectbox(
    "📝 Is it an Exam Day?",
    ["No", "Yes"]
)

food_type = st.selectbox(
    "🍚 Food Type",
    ["Rice_Dal", "Chole_Rice", "Rajma_Rice", "Poha", "Sandwich"]
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔮 PREDICT DEMAND", use_container_width=True):

    exam_value = 1 if exam_day == "Yes" else 0

    new_day = pd.DataFrame({
        "students_present": [students_present],
        "previous_sales": [previous_sales],
        "day": [day],
        "temperature": [temperature],
        "exam_day": [exam_value],
        "food_type": [food_type]
    })

    prediction = final_model.predict(new_day)[0]

    predicted_meals = round(prediction)

    safety_buffer = 5

    recommended_preparation = predicted_meals + safety_buffer


    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">🤖 AI Prediction</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Expected Meal Demand",
            f"{predicted_meals} meals"
        )

    with col2:
        st.metric(
            "Recommended Preparation",
            f"{recommended_preparation} meals"
        )


    st.success(
        f"The AI predicts a demand of approximately "
        f"{predicted_meals} meals."
    )

    st.info(
        f"A safety buffer of {safety_buffer} meals has been added, "
        f"giving a recommended preparation quantity of "
        f"{recommended_preparation} meals."
    )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📈 Model Performance</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "MAE",
        f"{mae:.2f} meals"
    )

with col2:
    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )


st.caption(
    "Linear Regression was selected after comparison with "
    "Random Forest Regression. Lower MAE and higher R² "
    "indicated better performance on the test data."
)


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

with st.expander("ℹ️ About this project"):

    st.write(
        "**Machine Learning Type:** Supervised Learning"
    )

    st.write(
        "**Problem Type:** Regression"
    )

    st.write(
        "**Target Variable:** Meals Sold"
    )

    st.write(
        "**Dataset Size:** 100 simulated records"
    )

    st.write(
        "**Training/Test Split:** 80% / 20%"
    )

    st.write(
        "**Models Compared:** Linear Regression and Random Forest Regression"
    )

    st.write(
        "**Selected Model:** Linear Regression"
    )


# --------------------------------------------------
# LIMITATION
# --------------------------------------------------

st.warning(
    "⚠️ Project limitation: The dataset used for this demonstration "
    "is simulated. Real school canteen data collected over a longer "
    "period would improve the reliability of predictions."
)
