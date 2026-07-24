"""
loan_default_ensemble.py
========================================================================
Loan Default Ensemble (v4) — Single-File Edition
------------------------------------------------------------------------
An advanced banking desktop application (PyQt5 + Scikit-learn + Pandas +
NumPy + Matplotlib + Seaborn + Joblib + ReportLab) that predicts loan
default risk using a Stacked Ensemble Model:

    Base Learner 1 : Random Forest Classifier
    Base Learner 2 : Gradient Boosting Classifier
    Meta Learner   : Logistic Regression (via StackingClassifier)

Run with:
    pip install -r requirements.txt
    python loan_default_ensemble.py

Tabs:
    1. Dashboard                 4. Applicant Form & Prediction
    2. Dataset & Preprocessing   5. Analytics Dashboard
    3. Model Training            6. Reports
========================================================================
"""

import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
)

import matplotlib
matplotlib.use("QT5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors as rl_colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFileDialog, QTableWidget,
    QTableWidgetItem, QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTextEdit, QMessageBox, QStatusBar, QLineEdit, QFrame,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont


# ==========================================================================
# SECTION 1: Constants / Utils
# ==========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "saved_model.pkl")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "loan_data.csv")

CATEGORICAL_COLUMNS = [
    "Gender", "Employment_Status", "Occupation", "Loan_Purpose",
    "Property_Ownership", "Marital_Status", "Payment_History",
]

BASE_NUMERIC_COLUMNS = [
    "Age", "Annual_Income", "Monthly_Income", "Loan_Amount", "Loan_Term",
    "Interest_Rate", "Credit_Score", "Existing_Debt",
    "Number_of_Credit_Cards", "Number_of_Previous_Loans", "Dependents",
    "Savings_Balance", "Checking_Balance", "Previous_Defaults",
    "Years_at_Current_Job",
]

ENGINEERED_COLUMNS = [
    "Debt_to_Income_Ratio", "Credit_Utilization", "Loan_to_Income_Ratio",
    "Employment_Stability_Score", "Financial_Risk_Index",
    "Payment_Reliability_Score",
]

FEATURE_COLUMNS = BASE_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ENGINEERED_COLUMNS
TARGET_COLUMN = "Loan_Default"

RISK_BANDS = [
    (0, 25, "Low Risk", "#2ecc71"),
    (25, 50, "Medium Risk", "#f1c40f"),
    (50, 75, "High Risk", "#e67e22"),
    (75, 100.0001, "Critical Risk", "#e74c3c"),
]

RECOMMENDATION_MAP = {
    "Low Risk": "Approve",
    "Medium Risk": "Review",
    "High Risk": "Review",
    "Critical Risk": "Reject",
}

APP_STYLE_DARK_BANK = """
QWidget { background-color: #10131c; color: #e6e6f0;
    font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }
QMainWindow { background-color: #10131c; }
QTabWidget::pane { border: 1px solid #232842; background: #161a29; }
QTabBar::tab { background: #1b2033; color: #cfd4ef; padding: 10px 16px;
    margin: 2px; border-radius: 6px; }
QTabBar::tab:selected { background: #c8a951; color: #10131c; font-weight: bold; }
QPushButton { background-color: #2f4d8a; color: white; border-radius: 6px;
    padding: 8px 14px; font-weight: 600; }
QPushButton:hover { background-color: #3c60ab; }
QPushButton:disabled { background-color: #2a2f45; color: #777; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #1b2033;
    border: 1px solid #2c3352; border-radius: 4px; padding: 5px; color: #f0f0f8; }
QTableWidget { background-color: #161a29; gridline-color: #2c3352; color: #e6e6f0; }
QHeaderView::section { background-color: #232842; color: #e6e6f0; padding: 4px; border: none; }
QProgressBar { border: 1px solid #2c3352; border-radius: 5px; text-align: center;
    color: white; background-color: #1b2033; }
QProgressBar::chunk { background-color: #c8a951; border-radius: 5px; }
QGroupBox { border: 1px solid #2c3352; border-radius: 6px; margin-top: 10px;
    padding-top: 10px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QLabel#TitleLabel { font-size: 22px; font-weight: bold; color: #c8a951; }
QLabel#SubtitleLabel { font-size: 13px; color: #9aa0c0; }
QStatusBar { background-color: #0c0e16; color: #9aa0c0; }
"""

DARK_BG, PANEL_BG, ACCENT, ACCENT2, TEXT_COLOR, GRID_COLOR = (
    "#10131c", "#161a29", "#c8a951", "#5c7fd6", "#e6e6f0", "#2c3352",
)


def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def format_currency(value):
    try:
        return f"${value:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def format_percentage(value):
    try:
        return f"{value:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def risk_category_for(probability_pct):
    """probability_pct expected in 0-100 range."""
    for lo, hi, label, color in RISK_BANDS:
        if lo <= probability_pct < hi:
            return label, color
    return "Critical Risk", RISK_BANDS[-1][3]


def generate_sample_dataset(path=DEFAULT_DATASET_PATH, n_rows=2000, seed=42):
    """Generates a synthetic loan applicant dataset for demo purposes."""
    rng = np.random.default_rng(seed)

    genders = ["Male", "Female", "Other"]
    employment = ["Employed", "Self-Employed", "Unemployed", "Retired"]
    occupations = ["Engineer", "Teacher", "Business Owner", "Clerk", "Doctor", "Driver", "Unemployed"]
    purposes = ["Home", "Auto", "Education", "Business", "Personal", "Medical"]
    property_status = ["Owned", "Mortgaged", "Rented"]
    marital = ["Single", "Married", "Divorced", "Widowed"]
    payment_hist = ["Excellent", "Good", "Fair", "Poor"]

    n = n_rows
    df = pd.DataFrame()
    df["Applicant_ID"] = np.arange(100000, 100000 + n)
    df["Age"] = rng.integers(21, 70, size=n)
    df["Gender"] = rng.choice(genders, size=n)
    df["Annual_Income"] = np.round(rng.uniform(15000, 220000, size=n), 2)
    df["Monthly_Income"] = np.round(df["Annual_Income"] / 12, 2)
    df["Employment_Status"] = rng.choice(employment, size=n, p=[0.65, 0.2, 0.08, 0.07])
    df["Occupation"] = rng.choice(occupations, size=n)
    df["Loan_Amount"] = np.round(rng.uniform(2000, 500000, size=n), 2)
    df["Loan_Term"] = rng.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 360], size=n)
    df["Interest_Rate"] = np.round(rng.uniform(3.5, 22.0, size=n), 2)
    df["Credit_Score"] = rng.integers(300, 851, size=n)
    df["Existing_Debt"] = np.round(rng.uniform(0, 150000, size=n), 2)
    df["Number_of_Credit_Cards"] = rng.integers(0, 10, size=n)
    df["Number_of_Previous_Loans"] = rng.integers(0, 8, size=n)
    df["Loan_Purpose"] = rng.choice(purposes, size=n)
    df["Property_Ownership"] = rng.choice(property_status, size=n)
    df["Marital_Status"] = rng.choice(marital, size=n)
    df["Dependents"] = rng.integers(0, 6, size=n)
    df["Savings_Balance"] = np.round(rng.uniform(0, 80000, size=n), 2)
    df["Checking_Balance"] = np.round(rng.uniform(0, 40000, size=n), 2)
    df["Payment_History"] = rng.choice(payment_hist, size=n, p=[0.3, 0.35, 0.2, 0.15])
    df["Previous_Defaults"] = rng.integers(0, 4, size=n)
    df["Years_at_Current_Job"] = rng.integers(0, 35, size=n)

    # Synthesize a realistic default probability from risk-correlated factors.
    # Coefficients and the sigmoid center/slope below were calibrated so the
    # resulting dataset has a realistic ~18-22% overall default rate rather
    # than an unrealistically high or low one.
    dti = (df["Existing_Debt"] / (df["Annual_Income"] + 1)).clip(upper=1.5)
    lti = (df["Loan_Amount"] / (df["Annual_Income"] + 1)).clip(upper=2.0)
    credit_risk = (750 - df["Credit_Score"]) / 450.0
    payment_penalty = df["Payment_History"].map(
        {"Excellent": -0.15, "Good": -0.05, "Fair": 0.1, "Poor": 0.3}
    )
    employment_penalty = df["Employment_Status"].map(
        {"Employed": -0.1, "Self-Employed": 0.0, "Retired": 0.05, "Unemployed": 0.35}
    )

    risk_score = (
        0.6 * dti
        + 0.35 * lti
        + 1.4 * credit_risk
        + 0.2 * df["Previous_Defaults"]
        + payment_penalty
        + employment_penalty
        + rng.normal(0, 0.15, size=n)
    )
    default_prob = 1 / (1 + np.exp(-2.8 * (risk_score - 2.6)))
    df["Loan_Default"] = (rng.uniform(0, 1, size=n) < default_prob).astype(int)

    return_path = path
    df.to_csv(return_path, index=False)
    return return_path


# ==========================================================================
# SECTION 2: Data Preprocessing
# ==========================================================================
class DataPreprocessor:
    """load -> clean -> engineer -> encode -> scale -> split, keeping
    encoders/scaler so the exact same transforms apply to a single
    applicant row entered in the GUI."""

    def __init__(self):
        self.raw_df = None
        self.processed_df = None
        self.label_encoders = {}
        self.scaler = None
        self.summary = {}

    # -------------------------------------------------------------- load --
    def load_csv(self, path):
        self.raw_df = pd.read_csv(path)
        return self.raw_df

    # ------------------------------------------------------------- clean --
    def clean(self, df=None):
        df = (self.raw_df if df is None else df).copy()
        n_before = len(df)
        n_duplicates = int(df.duplicated().sum())
        df = df.drop_duplicates()
        n_missing_before = int(df.isnull().sum().sum())

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

        # Outlier detection/clipping via IQR on key numeric columns
        outliers_clipped = 0
        for col in ["Annual_Income", "Loan_Amount", "Existing_Debt", "Savings_Balance", "Checking_Balance"]:
            if col in df.columns:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
                mask = (df[col] < lower) | (df[col] > upper)
                outliers_clipped += int(mask.sum())
                df[col] = df[col].clip(lower=lower, upper=upper)

        self.summary = {
            "rows_before": n_before,
            "rows_after_dedup": len(df),
            "duplicates_removed": n_duplicates,
            "missing_values_filled": n_missing_before,
            "outliers_clipped": outliers_clipped,
        }
        return df

    # -------------------------------------------------------- engineering --
    def engineer_features(self, df):
        df = df.copy()

        defaults = {
            "Annual_Income": 30000, "Existing_Debt": 0, "Loan_Amount": 0,
            "Credit_Score": 650, "Number_of_Credit_Cards": 1,
            "Years_at_Current_Job": 1, "Previous_Defaults": 0,
            "Payment_History": "Fair", "Employment_Status": "Employed",
        }
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val

        df["Debt_to_Income_Ratio"] = df["Existing_Debt"] / (df["Annual_Income"] + 1)

        credit_limit_proxy = (df["Number_of_Credit_Cards"].clip(lower=1)) * 5000
        df["Credit_Utilization"] = (df["Existing_Debt"] / credit_limit_proxy).clip(upper=5)

        df["Loan_to_Income_Ratio"] = df["Loan_Amount"] / (df["Annual_Income"] + 1)

        df["Employment_Stability_Score"] = (
            df["Years_at_Current_Job"] / (df["Years_at_Current_Job"] + 5)
        ) * 100

        credit_risk_component = (750 - df["Credit_Score"]).clip(lower=0) / 450.0
        df["Financial_Risk_Index"] = (
            0.4 * df["Debt_to_Income_Ratio"].clip(upper=3)
            + 0.3 * df["Loan_to_Income_Ratio"].clip(upper=5)
            + 0.3 * credit_risk_component
            + 0.1 * df["Previous_Defaults"]
        ) * 100

        payment_score_map = {"Excellent": 100, "Good": 75, "Fair": 50, "Poor": 20}
        base_payment_score = df["Payment_History"].map(payment_score_map).fillna(50)
        df["Payment_Reliability_Score"] = (
            base_payment_score - (df["Previous_Defaults"] * 10)
        ).clip(lower=0, upper=100)

        return df

    # -------------------------------------------------------------- encode --
    def encode(self, df):
        df = df.copy()
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        return df

    def encode_single_value(self, column, value):
        le = self.label_encoders.get(column)
        if le is None:
            return 0
        value = str(value)
        return int(le.transform([value])[0]) if value in le.classes_ else 0

    # --------------------------------------------------------------- scale --
    def fit_scaler(self, df):
        self.scaler = StandardScaler()
        self.scaler.fit(df[FEATURE_COLUMNS])
        return self.scaler

    def scale(self, df):
        df = df.copy()
        if self.scaler is None:
            self.fit_scaler(df)
        df[FEATURE_COLUMNS] = self.scaler.transform(df[FEATURE_COLUMNS])
        return df

    # ------------------------------------------------------------ pipeline --
    def run_pipeline(self, path):
        self.load_csv(path)
        df = self.clean(self.raw_df)
        df = self.engineer_features(df)

        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column (0/1 or Yes/No).")
        if df[TARGET_COLUMN].dtype == object:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).fillna(df[TARGET_COLUMN])
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

        df = self.encode(df)

        missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_features:
            raise ValueError(f"Dataset missing required columns: {missing_features}")

        self.processed_df = df  # unscaled copy kept for readable analytics charts
        return df

    def split(self, df=None, test_size=0.2, random_state=42):
        df = self.processed_df if df is None else df
        scaled_df = self.scale(df)
        X = scaled_df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def transform_single_row(self, inputs: dict) -> pd.DataFrame:
        """Applies the fitted encoders + scaler to a single applicant's raw inputs."""
        row = {}
        for col in BASE_NUMERIC_COLUMNS:
            row[col] = inputs.get(col, 0)
        for col in CATEGORICAL_COLUMNS:
            row[col] = self.encode_single_value(col, inputs.get(col))

        temp_df = pd.DataFrame([row])
        temp_df = self.engineer_features_from_raw(inputs, temp_df)

        for col in FEATURE_COLUMNS:
            if col not in temp_df.columns:
                temp_df[col] = 0

        temp_df = temp_df[FEATURE_COLUMNS]
        if self.scaler is not None:
            temp_df[FEATURE_COLUMNS] = self.scaler.transform(temp_df[FEATURE_COLUMNS])
        return temp_df

    @staticmethod
    def engineer_features_from_raw(inputs, temp_df):
        """Computes engineered features for a single raw applicant dict."""
        annual_income = inputs.get("Annual_Income", 30000) or 30000
        existing_debt = inputs.get("Existing_Debt", 0) or 0
        loan_amount = inputs.get("Loan_Amount", 0) or 0
        credit_score = inputs.get("Credit_Score", 650) or 650
        n_cards = max(inputs.get("Number_of_Credit_Cards", 1) or 1, 1)
        years_job = inputs.get("Years_at_Current_Job", 1) or 1
        previous_defaults = inputs.get("Previous_Defaults", 0) or 0
        payment_history = inputs.get("Payment_History", "Fair")

        dti = existing_debt / (annual_income + 1)
        credit_util = min(existing_debt / (n_cards * 5000), 5)
        lti = loan_amount / (annual_income + 1)
        emp_stability = (years_job / (years_job + 5)) * 100
        credit_risk_component = max(750 - credit_score, 0) / 450.0
        financial_risk_index = (
            0.4 * min(dti, 3) + 0.3 * min(lti, 5) + 0.3 * credit_risk_component
            + 0.1 * previous_defaults
        ) * 100
        payment_score_map = {"Excellent": 100, "Good": 75, "Fair": 50, "Poor": 20}
        base_payment_score = payment_score_map.get(payment_history, 50)
        payment_reliability = max(min(base_payment_score - previous_defaults * 10, 100), 0)

        temp_df["Debt_to_Income_Ratio"] = dti
        temp_df["Credit_Utilization"] = credit_util
        temp_df["Loan_to_Income_Ratio"] = lti
        temp_df["Employment_Stability_Score"] = emp_stability
        temp_df["Financial_Risk_Index"] = financial_risk_index
        temp_df["Payment_Reliability_Score"] = payment_reliability
        return temp_df


# ==========================================================================
# SECTION 3: Stacked Ensemble Model
# ==========================================================================
class EnsembleTrainer:
    """
    Wraps a Stacking ensemble of Random Forest + Gradient Boosting base
    learners with a Logistic Regression meta-learner, plus standalone
    copies of the base learners for model-comparison purposes.
    """

    def __init__(self, n_estimators_rf=300, n_estimators_gb=300,
                 max_depth_rf=8, max_depth_gb=4, random_state=42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators_rf, max_depth=max_depth_rf,
            random_state=random_state, n_jobs=-1,
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=n_estimators_gb, max_depth=max_depth_gb,
            random_state=random_state,
        )
        self.stack = StackingClassifier(
            estimators=[("rf", self.rf), ("gb", self.gb)],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5, n_jobs=-1,
        )
        self.metrics = {}
        self.individual_metrics = {}
        self.is_trained = False
        self.cv_scores = None

    def train(self, X_train, y_train, progress_callback=None):
        if progress_callback:
            progress_callback(10)
        self.stack.fit(X_train, y_train)
        if progress_callback:
            progress_callback(85)
        self.is_trained = True
        if progress_callback:
            progress_callback(100)
        return self.stack

    def evaluate(self, X_test, y_test):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
        y_pred = self.stack.predict(X_test)
        y_proba = self.stack.predict_proba(X_test)[:, 1]

        self.metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_proba),
        }
        cm = confusion_matrix(y_test, y_pred)
        return self.metrics, y_pred, y_proba, cm

    def compare_individual_models(self, X_train, y_train, X_test, y_test):
        """Fits standalone RF and GB models for accuracy comparison against
        the stacked ensemble."""
        rf_solo = RandomForestClassifier(
            n_estimators=self.rf.n_estimators, max_depth=self.rf.max_depth,
            random_state=42, n_jobs=-1,
        )
        gb_solo = GradientBoostingClassifier(
            n_estimators=self.gb.n_estimators, max_depth=self.gb.max_depth,
            random_state=42,
        )
        rf_solo.fit(X_train, y_train)
        gb_solo.fit(X_train, y_train)

        rf_acc = accuracy_score(y_test, rf_solo.predict(X_test))
        gb_acc = accuracy_score(y_test, gb_solo.predict(X_test))
        stack_acc = self.metrics.get("Accuracy") or accuracy_score(y_test, self.stack.predict(X_test))

        self.individual_metrics = {
            "Random Forest": rf_acc,
            "Gradient Boosting": gb_acc,
            "Stacked Ensemble": stack_acc,
        }
        return self.individual_metrics

    def cross_validate(self, X, y, cv=5):
        self.cv_scores = cross_val_score(self.stack, X, y, cv=cv, scoring="accuracy")
        return self.cv_scores

    def feature_importances(self):
        """Uses the Random Forest base learner's importances as a proxy
        for combined ensemble feature importance."""
        if not self.is_trained:
            return {}
        try:
            fitted_rf = self.stack.named_estimators_["rf"]
            return dict(zip(FEATURE_COLUMNS, fitted_rf.feature_importances_))
        except Exception:
            return {}

    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.stack.predict_proba(X)[:, 1]

    def save(self, path=DEFAULT_MODEL_PATH, extra=None):
        payload = {"stack": self.stack, "is_trained": self.is_trained}
        if extra:
            payload.update(extra)
        joblib.dump(payload, path)
        return path

    def load(self, path=DEFAULT_MODEL_PATH):
        payload = joblib.load(path)
        self.stack = payload["stack"]
        self.is_trained = payload.get("is_trained", True)
        return payload


# ==========================================================================
# SECTION 4: Predictor (single applicant risk assessment)
# ==========================================================================
class Predictor:
    """Bridges GUI applicant inputs -> encoded/scaled feature row ->
    ensemble probability -> risk category -> confidence -> recommendation."""

    def __init__(self, trainer, preprocessor):
        self.trainer = trainer
        self.preprocessor = preprocessor

    def predict(self, inputs: dict):
        X_row = self.preprocessor.transform_single_row(inputs)
        probability = float(self.trainer.predict_proba(X_row)[0]) * 100
        risk_label, risk_color = risk_category_for(probability)
        confidence = abs(probability - 50) * 2  # distance from decision boundary, scaled to 0-100
        recommendation = RECOMMENDATION_MAP.get(risk_label, "Review")
        return {
            "probability": probability,
            "risk_label": risk_label,
            "risk_color": risk_color,
            "confidence": round(confidence, 1),
            "recommendation": recommendation,
        }


# ==========================================================================
# SECTION 5: Visualization (Matplotlib + Seaborn chart builders)
# ==========================================================================
def _new_figure(figsize=(6.2, 4.2)):
    fig = Figure(figsize=figsize, dpi=100)
    fig.patch.set_facecolor(DARK_BG)
    return fig


def _style_axis(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def chart_feature_importance(importances: dict, top_n=12):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)
    items = sorted(importances.items(), key=lambda x: x[1])[-top_n:]
    ax.barh([i[0] for i in items], [i[1] for i in items], color=ACCENT2)
    ax.set_title("Feature Importance (Random Forest base learner)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def chart_roc_curve(y_test, y_proba):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    legend = ax.legend(loc="lower right", facecolor=PANEL_BG, edgecolor=GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    fig.tight_layout()
    return fig


def chart_precision_recall_curve(y_test, y_proba):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ax.plot(recall, precision, color=ACCENT2, linewidth=2)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    return fig


def chart_confusion_matrix(cm):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="mako", ax=ax, cbar=False,
        xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"],
    )
    ax.set_title("Confusion Matrix", color=TEXT_COLOR)
    ax.set_xlabel("Predicted", color=TEXT_COLOR)
    ax.set_ylabel("Actual", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    fig.patch.set_facecolor(DARK_BG)
    fig.tight_layout()
    return fig


def chart_risk_distribution(risk_labels_series):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)
    order = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    colors_map = {label: color for _, _, label, color in RISK_BANDS}
    counts = risk_labels_series.value_counts().reindex(order).fillna(0)
    ax.bar(counts.index, counts.values, color=[colors_map[o] for o in order])
    ax.set_title("Risk Category Distribution")
    ax.set_ylabel("Number of Applicants")
    fig.tight_layout()
    return fig


def chart_default_rate_by_income(df, income_col="Annual_Income", target_col=TARGET_COLUMN, bins=8):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)
    temp = df[[income_col, target_col]].copy()
    temp["income_bin"] = pd.qcut(temp[income_col], q=bins, duplicates="drop")
    grouped = temp.groupby("income_bin", observed=True)[target_col].mean() * 100
    labels = [f"{int(iv.left/1000)}k-{int(iv.right/1000)}k" for iv in grouped.index]
    ax.bar(labels, grouped.values, color=ACCENT)
    ax.set_title("Default Rate by Income Bracket")
    ax.set_ylabel("Default Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def chart_credit_score_distribution(df, col="Credit_Score"):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)
    ax.hist(df[col], bins=30, color=ACCENT2, edgecolor=DARK_BG)
    ax.set_title("Credit Score Distribution")
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def chart_loan_amount_distribution(df, col="Loan_Amount"):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)
    ax.hist(df[col], bins=30, color=ACCENT, edgecolor=DARK_BG)
    ax.set_title("Loan Amount Distribution")
    ax.set_xlabel("Loan Amount")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def chart_correlation_heatmap(df):
    fig = _new_figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", ax=ax, cbar=True, center=0,
                xticklabels=True, yticklabels=True)
    ax.set_title("Correlation Heatmap", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=6)
    fig.patch.set_facecolor(DARK_BG)
    fig.tight_layout()
    return fig


# ==========================================================================
# SECTION 6: Background training thread (keeps GUI responsive)
# ==========================================================================
class TrainingWorker(QThread):
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, trainer, X_train, X_test, y_train, y_test):
        super().__init__()
        self.trainer = trainer
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def run(self):
        try:
            self.trainer.train(self.X_train, self.y_train, progress_callback=self.progress.emit)
            metrics, y_pred, y_proba, cm = self.trainer.evaluate(self.X_test, self.y_test)
            comparison = self.trainer.compare_individual_models(
                self.X_train, self.y_train, self.X_test, self.y_test
            )
            cv_scores = self.trainer.cross_validate(
                pd.concat([self.X_train, self.X_test]),
                pd.concat([self.y_train, self.y_test]),
                cv=5,
            )
            self.finished_ok.emit({
                "metrics": metrics, "y_pred": y_pred, "y_proba": y_proba,
                "cm": cm, "comparison": comparison, "cv_scores": cv_scores,
            })
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


# ==========================================================================
# SECTION 7: Main Window (PyQt5 multi-tab banking GUI)
# ==========================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.setWindowTitle("Loan Default Ensemble v4 — Banking Risk Decision Support System")
        self.resize(1360, 860)
        self.setStyleSheet(APP_STYLE_DARK_BANK)

        self.preprocessor = DataPreprocessor()
        self.trainer = EnsembleTrainer()
        self.predictor = None
        self.dataset_path = None
        self.processed_df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_pred = self.y_proba = self.cm = None
        self.training_worker = None
        self.prediction_history = []
        self.last_prediction = None

        self._build_ui()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_dashboard_tab(), "Dashboard")
        self.tabs.addTab(self._build_dataset_tab(), "Dataset && Preprocessing")
        self.tabs.addTab(self._build_training_tab(), "Model Training")
        self.tabs.addTab(self._build_prediction_tab(), "Applicant Form && Prediction")
        self.tabs.addTab(self._build_analytics_tab(), "Analytics Dashboard")
        self.tabs.addTab(self._build_reports_tab(), "Reports")

        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load an applicant dataset to begin.")

    # ---------------------------------------------------- Tab 1: Dashboard
    def _build_dashboard_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title = QLabel("Loan Default Ensemble — Banking Risk Dashboard")
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "Stacked Ensemble (Random Forest + Gradient Boosting, Logistic Regression meta-learner)\n"
            "for loan default probability scoring and risk-based decisioning."
        )
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(15)

        grid = QGridLayout()
        self.dash_total_label = QLabel("Total Applicants: —")
        self.dash_approved_label = QLabel("Approved Loans: —")
        self.dash_default_label = QLabel("Default Cases: —")
        self.dash_avg_risk_label = QLabel("Average Risk Score: —")
        self.dash_accuracy_label = QLabel("Model Accuracy: —")
        self.dash_model_status_label = QLabel("Model status: Not trained")

        for i, lbl in enumerate([
            self.dash_total_label, self.dash_approved_label, self.dash_default_label,
            self.dash_avg_risk_label, self.dash_accuracy_label, self.dash_model_status_label,
        ]):
            box = QGroupBox()
            box_layout = QVBoxLayout(box)
            lbl.setFont(QFont("Segoe UI", 11))
            box_layout.addWidget(lbl)
            grid.addWidget(box, i // 3, i % 3)
        layout.addLayout(grid)

        info_box = QGroupBox("System Overview")
        info_layout = QVBoxLayout(info_box)
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setPlainText(
            "Workflow:\n"
            "  1. Load Dataset -> 2. Clean/Engineer Features -> 3. Label Encode -> 4. Scale\n"
            "  5. Train/Test Split -> 6. Train Random Forest -> 7. Train Gradient Boosting\n"
            "  8. Stack Both Models -> 9. Generate Final Prediction -> 10. Evaluate -> 11. Display Results\n\n"
            "Base Learners: Random Forest Classifier, Gradient Boosting Classifier\n"
            "Meta Learner: Logistic Regression (via StackingClassifier)\n"
            "Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Cross-Validation\n"
            "Risk Bands: 0-25% Low | 26-50% Medium | 51-75% High | 76-100% Critical"
        )
        info_layout.addWidget(info_text)
        layout.addWidget(info_box)
        return widget

    def _refresh_dashboard(self):
        if self.processed_df is not None:
            total = len(self.processed_df)
            defaults = int(self.processed_df[TARGET_COLUMN].sum())
            approved = total - defaults
            self.dash_total_label.setText(f"Total Applicants: {total}")
            self.dash_approved_label.setText(f"Approved Loans: {approved}")
            self.dash_default_label.setText(f"Default Cases: {defaults}")

        if self.prediction_history:
            avg_risk = np.mean([p["probability"] for p in self.prediction_history])
            self.dash_avg_risk_label.setText(f"Average Risk Score: {avg_risk:.1f}%")

        if self.trainer.is_trained:
            self.dash_model_status_label.setText("Model status: Trained ✔")
        if self.trainer.metrics:
            self.dash_accuracy_label.setText(f"Model Accuracy: {self.trainer.metrics['Accuracy']*100:.2f}%")

    # ------------------------------------------- Tab 2: Dataset & Preprocessing
    def _build_dataset_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_row = QHBoxLayout()
        browse_btn = QPushButton("Browse CSV…")
        browse_btn.clicked.connect(self._on_browse_csv)
        sample_btn = QPushButton("Generate Sample Dataset")
        sample_btn.clicked.connect(self._on_generate_sample)
        preprocess_btn = QPushButton("Run Preprocessing Pipeline")
        preprocess_btn.clicked.connect(self._on_run_preprocessing)
        btn_row.addWidget(browse_btn)
        btn_row.addWidget(sample_btn)
        btn_row.addWidget(preprocess_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.dataset_path_label = QLabel("No dataset loaded.")
        layout.addWidget(self.dataset_path_label)

        self.dataset_stats_label = QLabel("")
        self.dataset_stats_label.setWordWrap(True)
        layout.addWidget(self.dataset_stats_label)

        self.preprocess_summary = QTextEdit()
        self.preprocess_summary.setReadOnly(True)
        self.preprocess_summary.setMaximumHeight(150)
        layout.addWidget(self.preprocess_summary)

        self.dataset_table = QTableWidget()
        layout.addWidget(self.dataset_table)
        return widget

    def _on_browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Loan Applicant CSV", "", "CSV Files (*.csv)")
        if path:
            self._load_dataset(path)

    def _on_generate_sample(self):
        self._load_dataset(generate_sample_dataset())

    def _load_dataset(self, path):
        try:
            df = pd.read_csv(path)
            self.dataset_path = path
            self.dataset_path_label.setText(f"Loaded: {path}")
            self.dataset_stats_label.setText(
                f"Rows: {len(df)}  |  Columns: {len(df.columns)}  |  "
                f"Missing values: {int(df.isnull().sum().sum())}  |  "
                f"Duplicate rows: {int(df.duplicated().sum())}"
            )
            self._populate_table(self.dataset_table, df.head(100))
            self.status_bar.showMessage(f"Dataset loaded: {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error Loading Dataset", str(exc))

    def _on_run_preprocessing(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset first.")
            return
        try:
            df = self.preprocessor.run_pipeline(self.dataset_path)
            self.processed_df = df
            summary = self.preprocessor.summary
            encoded_cols = list(self.preprocessor.label_encoders.keys())
            text = (
                f"Rows before cleaning: {summary.get('rows_before')}\n"
                f"Duplicates removed: {summary.get('duplicates_removed')}\n"
                f"Rows after dedup: {summary.get('rows_after_dedup')}\n"
                f"Missing values filled: {summary.get('missing_values_filled')}\n"
                f"Outliers clipped: {summary.get('outliers_clipped')}\n"
                f"Encoded categorical columns: {encoded_cols}\n"
                f"Engineered features: {ENGINEERED_COLUMNS}\n"
                f"Final processed shape: {df.shape}"
            )
            self.preprocess_summary.setPlainText(text)
            self._populate_table(self.dataset_table, df.head(100))
            self.status_bar.showMessage("Preprocessing complete.")
            self._refresh_dashboard()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Preprocessing Error", str(exc))
            traceback.print_exc()

    @staticmethod
    def _populate_table(table: QTableWidget, df: pd.DataFrame):
        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))
        table.resizeColumnsToContents()

    # -------------------------------------------------- Tab 3: Model Training
    def _build_training_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        params_box = QGroupBox("Ensemble Training Parameters")
        params_layout = QGridLayout(params_box)

        params_layout.addWidget(QLabel("RF Estimators:"), 0, 0)
        self.spin_rf_estimators = QSpinBox()
        self.spin_rf_estimators.setRange(10, 2000)
        self.spin_rf_estimators.setValue(300)
        params_layout.addWidget(self.spin_rf_estimators, 0, 1)

        params_layout.addWidget(QLabel("GB Estimators:"), 0, 2)
        self.spin_gb_estimators = QSpinBox()
        self.spin_gb_estimators.setRange(10, 2000)
        self.spin_gb_estimators.setValue(300)
        params_layout.addWidget(self.spin_gb_estimators, 0, 3)

        params_layout.addWidget(QLabel("RF Max Depth:"), 1, 0)
        self.spin_rf_depth = QSpinBox()
        self.spin_rf_depth.setRange(1, 30)
        self.spin_rf_depth.setValue(8)
        params_layout.addWidget(self.spin_rf_depth, 1, 1)

        params_layout.addWidget(QLabel("GB Max Depth:"), 1, 2)
        self.spin_gb_depth = QSpinBox()
        self.spin_gb_depth.setRange(1, 20)
        self.spin_gb_depth.setValue(4)
        params_layout.addWidget(self.spin_gb_depth, 1, 3)

        params_layout.addWidget(QLabel("Test Split %:"), 2, 0)
        self.spin_test_split = QSpinBox()
        self.spin_test_split.setRange(5, 50)
        self.spin_test_split.setValue(20)
        params_layout.addWidget(self.spin_test_split, 2, 1)

        layout.addWidget(params_box)

        btn_row = QHBoxLayout()
        self.train_btn = QPushButton("Train Stacked Ensemble")
        self.train_btn.clicked.connect(self._on_train_model)
        save_model_btn = QPushButton("Save Model (.pkl)")
        save_model_btn.clicked.connect(self._on_save_model)
        load_model_btn = QPushButton("Load Existing Model")
        load_model_btn.clicked.connect(self._on_load_model)
        btn_row.addWidget(self.train_btn)
        btn_row.addWidget(save_model_btn)
        btn_row.addWidget(load_model_btn)
        layout.addLayout(btn_row)

        self.train_progress = QProgressBar()
        layout.addWidget(self.train_progress)

        self.metrics_label = QTextEdit()
        self.metrics_label.setReadOnly(True)
        layout.addWidget(self.metrics_label)
        return widget

    def _on_train_model(self):
        if self.processed_df is None:
            QMessageBox.warning(self, "No Processed Data", "Please run preprocessing first.")
            return

        self.trainer = EnsembleTrainer(
            n_estimators_rf=self.spin_rf_estimators.value(),
            n_estimators_gb=self.spin_gb_estimators.value(),
            max_depth_rf=self.spin_rf_depth.value(),
            max_depth_gb=self.spin_gb_depth.value(),
        )
        test_size = self.spin_test_split.value() / 100.0
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.split(
            self.processed_df, test_size=test_size
        )

        self.train_btn.setEnabled(False)
        self.train_progress.setValue(0)
        self.status_bar.showMessage("Training stacked ensemble… this may take a moment.")

        self.training_worker = TrainingWorker(
            self.trainer, self.X_train, self.X_test, self.y_train, self.y_test
        )
        self.training_worker.progress.connect(self.train_progress.setValue)
        self.training_worker.finished_ok.connect(self._on_training_finished)
        self.training_worker.failed.connect(self._on_training_failed)
        self.training_worker.start()

    def _on_training_finished(self, result):
        self.train_btn.setEnabled(True)
        metrics = result["metrics"]
        self.y_pred, self.y_proba, self.cm = result["y_pred"], result["y_proba"], result["cm"]
        comparison = result["comparison"]
        cv_scores = result["cv_scores"]
        self.predictor = Predictor(self.trainer, self.preprocessor)

        text = (
            f"Training complete.\n\n"
            f"Accuracy:  {metrics['Accuracy']*100:.2f}%\n"
            f"Precision: {metrics['Precision']*100:.2f}%\n"
            f"Recall:    {metrics['Recall']*100:.2f}%\n"
            f"F1 Score:  {metrics['F1']*100:.2f}%\n"
            f"ROC-AUC:   {metrics['ROC_AUC']:.4f}\n\n"
            f"Cross-Validation Accuracy (5-fold): {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)\n\n"
            f"Model Comparison (Test Accuracy):\n"
            f"  Random Forest:      {comparison['Random Forest']*100:.2f}%\n"
            f"  Gradient Boosting:  {comparison['Gradient Boosting']*100:.2f}%\n"
            f"  Stacked Ensemble:   {comparison['Stacked Ensemble']*100:.2f}%\n"
        )
        self.metrics_label.setPlainText(text)
        self.status_bar.showMessage("Model trained successfully.")
        self._refresh_dashboard()

    def _on_training_failed(self, error_msg):
        self.train_btn.setEnabled(True)
        QMessageBox.critical(self, "Training Failed", error_msg)
        self.status_bar.showMessage("Training failed.")

    def _on_save_model(self):
        if not self.trainer.is_trained:
            QMessageBox.warning(self, "Model Not Trained", "Train a model before saving.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", DEFAULT_MODEL_PATH, "Pickle Files (*.pkl)")
        if path:
            self.trainer.save(path, extra={
                "label_encoders": self.preprocessor.label_encoders,
                "scaler": self.preprocessor.scaler,
            })
            QMessageBox.information(self, "Model Saved", f"Model saved to:\n{path}")
            self.status_bar.showMessage(f"Model saved to {path}")

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", DEFAULT_MODEL_PATH, "Pickle Files (*.pkl)")
        if path:
            try:
                payload = self.trainer.load(path)
                if "label_encoders" in payload:
                    self.preprocessor.label_encoders = payload["label_encoders"]
                if "scaler" in payload:
                    self.preprocessor.scaler = payload["scaler"]
                self.predictor = Predictor(self.trainer, self.preprocessor)
                QMessageBox.information(self, "Model Loaded", f"Model loaded from:\n{path}")
                self.status_bar.showMessage(f"Model loaded from {path}")
                self._refresh_dashboard()
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Load Error", str(exc))

    # ----------------------------------------- Tab 4: Applicant Form & Prediction
    def _build_prediction_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # ---- Left: Applicant Form ----
        form_box = QGroupBox("Applicant Information")
        form_layout = QGridLayout(form_box)

        self.f_name = QLineEdit()
        self.f_age = QSpinBox(); self.f_age.setRange(18, 100); self.f_age.setValue(35)
        self.f_gender = QComboBox(); self.f_gender.addItems(["Male", "Female", "Other"])
        self.f_income = QDoubleSpinBox(); self.f_income.setRange(0, 10_000_000); self.f_income.setValue(60000)
        self.f_employment = QComboBox(); self.f_employment.addItems(
            ["Employed", "Self-Employed", "Unemployed", "Retired"])
        self.f_loan_amount = QDoubleSpinBox(); self.f_loan_amount.setRange(0, 10_000_000); self.f_loan_amount.setValue(20000)
        self.f_loan_term = QSpinBox(); self.f_loan_term.setRange(1, 480); self.f_loan_term.setValue(60)
        self.f_interest_rate = QDoubleSpinBox(); self.f_interest_rate.setRange(0, 100); self.f_interest_rate.setValue(9.5)
        self.f_credit_score = QSpinBox(); self.f_credit_score.setRange(300, 850); self.f_credit_score.setValue(680)
        self.f_existing_debt = QDoubleSpinBox(); self.f_existing_debt.setRange(0, 10_000_000); self.f_existing_debt.setValue(5000)
        self.f_credit_cards = QSpinBox(); self.f_credit_cards.setRange(0, 20); self.f_credit_cards.setValue(2)
        self.f_previous_loans = QSpinBox(); self.f_previous_loans.setRange(0, 20); self.f_previous_loans.setValue(1)
        self.f_loan_purpose = QComboBox(); self.f_loan_purpose.addItems(
            ["Home", "Auto", "Education", "Business", "Personal", "Medical"])
        self.f_property = QComboBox(); self.f_property.addItems(["Owned", "Mortgaged", "Rented"])
        self.f_marital = QComboBox(); self.f_marital.addItems(["Single", "Married", "Divorced", "Widowed"])
        self.f_dependents = QSpinBox(); self.f_dependents.setRange(0, 15)
        self.f_savings = QDoubleSpinBox(); self.f_savings.setRange(0, 10_000_000); self.f_savings.setValue(5000)
        self.f_checking = QDoubleSpinBox(); self.f_checking.setRange(0, 10_000_000); self.f_checking.setValue(2000)
        self.f_payment_history = QComboBox(); self.f_payment_history.addItems(
            ["Excellent", "Good", "Fair", "Poor"])
        self.f_previous_defaults = QSpinBox(); self.f_previous_defaults.setRange(0, 10)
        self.f_years_job = QSpinBox(); self.f_years_job.setRange(0, 50); self.f_years_job.setValue(4)

        fields = [
            ("Name", self.f_name), ("Age", self.f_age), ("Gender", self.f_gender),
            ("Annual Income", self.f_income), ("Employment Status", self.f_employment),
            ("Loan Amount", self.f_loan_amount), ("Loan Term (months)", self.f_loan_term),
            ("Interest Rate (%)", self.f_interest_rate), ("Credit Score", self.f_credit_score),
            ("Existing Debt", self.f_existing_debt), ("Number of Credit Cards", self.f_credit_cards),
            ("Number of Previous Loans", self.f_previous_loans), ("Loan Purpose", self.f_loan_purpose),
            ("Property Ownership", self.f_property), ("Marital Status", self.f_marital),
            ("Dependents", self.f_dependents), ("Savings Balance", self.f_savings),
            ("Checking Balance", self.f_checking), ("Payment History", self.f_payment_history),
            ("Previous Defaults", self.f_previous_defaults), ("Years at Current Job", self.f_years_job),
        ]
        for i, (label_text, field) in enumerate(fields):
            form_layout.addWidget(QLabel(label_text), i, 0)
            form_layout.addWidget(field, i, 1)

        btn_row = QHBoxLayout()
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self._on_predict)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear_form)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save_prediction)
        btn_row.addWidget(predict_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(save_btn)
        form_layout.addLayout(btn_row, len(fields), 0, 1, 2)

        # ---- Right: Prediction Result ----
        result_box = QGroupBox("Prediction Result")
        result_layout = QVBoxLayout(result_box)

        self.result_probability_label = QLabel("Default Probability: —")
        self.result_probability_label.setObjectName("TitleLabel")
        self.result_risk_label = QLabel("Risk Category: —")
        self.result_risk_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.result_confidence_label = QLabel("Confidence Score: —")
        self.result_recommendation_label = QLabel("Recommendation: —")
        self.result_recommendation_label.setFont(QFont("Segoe UI", 13, QFont.Bold))

        for lbl in [self.result_probability_label, self.result_risk_label,
                    self.result_confidence_label, self.result_recommendation_label]:
            result_layout.addWidget(lbl)

        self.risk_color_bar = QFrame()
        self.risk_color_bar.setFixedHeight(24)
        self.risk_color_bar.setStyleSheet("background-color: #444;")
        result_layout.addWidget(self.risk_color_bar)
        result_layout.addStretch()

        # Wrap form in a scroll-free container with fixed proportional widths
        layout.addWidget(form_box, 3)
        layout.addWidget(result_box, 2)
        return widget

    def _collect_form_inputs(self):
        return {
            "Age": self.f_age.value(),
            "Gender": self.f_gender.currentText(),
            "Annual_Income": self.f_income.value(),
            "Monthly_Income": self.f_income.value() / 12.0,
            "Employment_Status": self.f_employment.currentText(),
            "Occupation": "Unknown",
            "Loan_Amount": self.f_loan_amount.value(),
            "Loan_Term": self.f_loan_term.value(),
            "Interest_Rate": self.f_interest_rate.value(),
            "Credit_Score": self.f_credit_score.value(),
            "Existing_Debt": self.f_existing_debt.value(),
            "Number_of_Credit_Cards": self.f_credit_cards.value(),
            "Number_of_Previous_Loans": self.f_previous_loans.value(),
            "Loan_Purpose": self.f_loan_purpose.currentText(),
            "Property_Ownership": self.f_property.currentText(),
            "Marital_Status": self.f_marital.currentText(),
            "Dependents": self.f_dependents.value(),
            "Savings_Balance": self.f_savings.value(),
            "Checking_Balance": self.f_checking.value(),
            "Payment_History": self.f_payment_history.currentText(),
            "Previous_Defaults": self.f_previous_defaults.value(),
            "Years_at_Current_Job": self.f_years_job.value(),
        }

    def _on_predict(self):
        if self.predictor is None:
            QMessageBox.warning(self, "Model Not Ready", "Train or load a model first.")
            return
        inputs = self._collect_form_inputs()
        try:
            result = self.predictor.predict(inputs)
            result["name"] = self.f_name.text() or "Unnamed Applicant"
            result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result["inputs"] = inputs
            self.last_prediction = result

            self.result_probability_label.setText(f"Default Probability: {result['probability']:.2f}%")
            self.result_risk_label.setText(f"Risk Category: {result['risk_label']}")
            self.result_risk_label.setStyleSheet(f"color: {result['risk_color']};")
            self.result_confidence_label.setText(f"Confidence Score: {result['confidence']}%")
            self.result_recommendation_label.setText(f"Recommendation: {result['recommendation']}")
            self.risk_color_bar.setStyleSheet(f"background-color: {result['risk_color']};")

            self.status_bar.showMessage("Prediction generated.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Prediction Error", str(exc))

    def _on_clear_form(self):
        self.f_name.clear()
        self.f_age.setValue(35)
        self.f_income.setValue(60000)
        self.f_loan_amount.setValue(20000)
        self.f_loan_term.setValue(60)
        self.f_interest_rate.setValue(9.5)
        self.f_credit_score.setValue(680)
        self.f_existing_debt.setValue(5000)
        self.f_credit_cards.setValue(2)
        self.f_previous_loans.setValue(1)
        self.f_dependents.setValue(0)
        self.f_savings.setValue(5000)
        self.f_checking.setValue(2000)
        self.f_previous_defaults.setValue(0)
        self.f_years_job.setValue(4)
        self.result_probability_label.setText("Default Probability: —")
        self.result_risk_label.setText("Risk Category: —")
        self.result_risk_label.setStyleSheet("")
        self.result_confidence_label.setText("Confidence Score: —")
        self.result_recommendation_label.setText("Recommendation: —")
        self.risk_color_bar.setStyleSheet("background-color: #444;")

    def _on_save_prediction(self):
        if not self.last_prediction:
            QMessageBox.warning(self, "No Prediction", "Run a prediction before saving.")
            return
        self.prediction_history.append(self.last_prediction)
        self._refresh_dashboard()
        QMessageBox.information(self, "Saved", "Prediction saved to history.")
        self.status_bar.showMessage("Prediction saved to history.")

    # ------------------------------------------------- Tab 5: Analytics
    def _build_analytics_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_row = QHBoxLayout()
        self.chart_selector = QComboBox()
        self.chart_selector.addItems([
            "Feature Importance", "ROC Curve", "Precision-Recall Curve",
            "Confusion Matrix", "Risk Distribution", "Default Rate by Income",
            "Credit Score Distribution", "Loan Amount Distribution",
            "Correlation Heatmap",
        ])
        render_btn = QPushButton("Render Chart")
        render_btn.clicked.connect(self._on_render_chart)
        btn_row.addWidget(self.chart_selector)
        btn_row.addWidget(render_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.chart_container_layout = QVBoxLayout()
        self.chart_widget_placeholder = QLabel("Select a chart and click 'Render Chart'.")
        self.chart_container_layout.addWidget(self.chart_widget_placeholder)
        layout.addLayout(self.chart_container_layout)

        self.current_canvas = None
        return widget

    def _clear_chart_area(self):
        if self.current_canvas is not None:
            self.chart_container_layout.removeWidget(self.current_canvas)
            self.current_canvas.setParent(None)
            self.current_canvas = None
        else:
            self.chart_container_layout.removeWidget(self.chart_widget_placeholder)
            self.chart_widget_placeholder.setParent(None)

    def _on_render_chart(self):
        choice = self.chart_selector.currentText()
        try:
            if choice == "Feature Importance":
                if not self.trainer.is_trained:
                    raise ValueError("Train the model first.")
                fig = chart_feature_importance(self.trainer.feature_importances())
            elif choice == "ROC Curve":
                if self.y_proba is None:
                    raise ValueError("Train the model first.")
                fig = chart_roc_curve(self.y_test, self.y_proba)
            elif choice == "Precision-Recall Curve":
                if self.y_proba is None:
                    raise ValueError("Train the model first.")
                fig = chart_precision_recall_curve(self.y_test, self.y_proba)
            elif choice == "Confusion Matrix":
                if self.cm is None:
                    raise ValueError("Train the model first.")
                fig = chart_confusion_matrix(self.cm)
            elif choice == "Risk Distribution":
                if not self.prediction_history:
                    raise ValueError("Save at least one prediction first (Applicant Form tab).")
                risk_series = pd.Series([p["risk_label"] for p in self.prediction_history])
                fig = chart_risk_distribution(risk_series)
            elif choice == "Default Rate by Income":
                if self.processed_df is None:
                    raise ValueError("Load and preprocess a dataset first.")
                fig = chart_default_rate_by_income(self.processed_df)
            elif choice == "Credit Score Distribution":
                if self.processed_df is None:
                    raise ValueError("Load and preprocess a dataset first.")
                fig = chart_credit_score_distribution(self.processed_df)
            elif choice == "Loan Amount Distribution":
                if self.processed_df is None:
                    raise ValueError("Load and preprocess a dataset first.")
                fig = chart_loan_amount_distribution(self.processed_df)
            elif choice == "Correlation Heatmap":
                if self.processed_df is None:
                    raise ValueError("Load and preprocess a dataset first.")
                fig = chart_correlation_heatmap(self.processed_df)
            else:
                return

            self._clear_chart_area()
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(480)
            self.chart_container_layout.addWidget(canvas)
            self.current_canvas = canvas
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Cannot Render Chart", str(exc))

    # ------------------------------------------------------ Tab 6: Reports
    def _build_reports_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_row = QHBoxLayout()
        export_csv_btn = QPushButton("Export Predictions (CSV)")
        export_csv_btn.clicked.connect(self._on_export_predictions_csv)
        export_excel_btn = QPushButton("Export Scorecard (Excel)")
        export_excel_btn.clicked.connect(self._on_export_scorecard_excel)
        export_pdf_btn = QPushButton("Export Model Evaluation (PDF)")
        export_pdf_btn.clicked.connect(self._on_export_evaluation_pdf)
        btn_row.addWidget(export_csv_btn)
        btn_row.addWidget(export_excel_btn)
        btn_row.addWidget(export_pdf_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        layout.addWidget(self.report_preview)
        return widget

    def _on_export_predictions_csv(self):
        if not self.prediction_history:
            QMessageBox.warning(self, "No Predictions", "Save at least one prediction first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Predictions", os.path.join(REPORTS_DIR, "prediction_results.csv"), "CSV Files (*.csv)"
        )
        if path:
            rows = []
            for p in self.prediction_history:
                row = {"Name": p["name"], "Timestamp": p["timestamp"],
                       "Default_Probability_%": round(p["probability"], 2),
                       "Risk_Category": p["risk_label"], "Confidence_%": p["confidence"],
                       "Recommendation": p["recommendation"]}
                row.update(p["inputs"])
                rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            QMessageBox.information(self, "Exported", f"Predictions exported to:\n{path}")

    def _on_export_scorecard_excel(self):
        if not self.last_prediction:
            QMessageBox.warning(self, "No Prediction", "Run a prediction first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Applicant Scorecard", os.path.join(REPORTS_DIR, "applicant_scorecard.xlsx"),
            "Excel Files (*.xlsx)",
        )
        if not path:
            return
        p = self.last_prediction
        scorecard = {
            "Field": ["Name", "Timestamp", "Default Probability (%)", "Risk Category",
                      "Confidence Score (%)", "Recommendation"] + list(p["inputs"].keys()),
            "Value": [p["name"], p["timestamp"], round(p["probability"], 2), p["risk_label"],
                      p["confidence"], p["recommendation"]] + list(p["inputs"].values()),
        }
        pd.DataFrame(scorecard).to_excel(path, index=False, sheet_name="Scorecard")
        QMessageBox.information(self, "Exported", f"Scorecard exported to:\n{path}")

    def _on_export_evaluation_pdf(self):
        if not self.trainer.metrics:
            QMessageBox.warning(self, "No Metrics", "Train the model to generate evaluation metrics first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Model Evaluation Report", os.path.join(REPORTS_DIR, "model_evaluation_report.pdf"),
            "PDF Files (*.pdf)",
        )
        if not path:
            return
        try:
            doc = SimpleDocTemplate(path, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = [
                Paragraph("Loan Default Ensemble — Model Evaluation Report", styles["Title"]),
                Spacer(1, 12),
                Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
                Spacer(1, 12),
            ]

            metrics_data = [["Metric", "Value"]] + [
                [k, f"{v:.4f}"] for k, v in self.trainer.metrics.items()
            ]
            metrics_table = Table(metrics_data, hAlign="LEFT")
            metrics_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2f4d8a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]))
            elements.append(Paragraph("Evaluation Metrics", styles["Heading2"]))
            elements.append(metrics_table)
            elements.append(Spacer(1, 16))

            if self.trainer.individual_metrics:
                comp_data = [["Model", "Test Accuracy"]] + [
                    [k, f"{v*100:.2f}%"] for k, v in self.trainer.individual_metrics.items()
                ]
                comp_table = Table(comp_data, hAlign="LEFT")
                comp_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#c8a951")),
                    ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                ]))
                elements.append(Paragraph("Model Comparison", styles["Heading2"]))
                elements.append(comp_table)
                elements.append(Spacer(1, 16))

            if self.cm is not None:
                cm_data = [["", "Predicted No Default", "Predicted Default"],
                           ["Actual No Default", str(self.cm[0][0]), str(self.cm[0][1])],
                           ["Actual Default", str(self.cm[1][0]), str(self.cm[1][1])]]
                cm_table = Table(cm_data, hAlign="LEFT")
                cm_table.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                ]))
                elements.append(Paragraph("Confusion Matrix", styles["Heading2"]))
                elements.append(cm_table)

            doc.build(elements)
            QMessageBox.information(self, "Exported", f"Evaluation report exported to:\n{path}")
            self.report_preview.setPlainText(f"PDF report saved to: {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export Error", str(exc))


# ==========================================================================
# SECTION 8: Entry Point
# ==========================================================================
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
