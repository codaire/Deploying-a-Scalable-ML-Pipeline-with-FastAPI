import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


@pytest.fixture
def sample_data():
    """Small synthetic dataset matching the census feature schema."""
    data = pd.DataFrame({
        "age": [39, 50, 38, 53, 28],
        "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private", "Private"],
        "fnlgt": [77516, 83311, 215646, 234721, 338409],
        "education": ["Bachelors", "Bachelors", "HS-grad", "11th", "Bachelors"],
        "education-num": [13, 13, 9, 7, 13],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced",
                           "Married-civ-spouse", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial", "Handlers-cleaners",
                       "Handlers-cleaners", "Prof-specialty"],
        "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband", "Wife"],
        "race": ["White", "White", "White", "Black", "Black"],
        "sex": ["Male", "Male", "Male", "Male", "Female"],
        "capital-gain": [2174, 0, 0, 0, 0],
        "capital-loss": [0, 0, 0, 0, 0],
        "hours-per-week": [40, 13, 40, 40, 40],
        "native-country": ["United-States"] * 5,
        "salary": [">50K", "<=50K", "<=50K", "<=50K", "<=50K"],
    })
    return data


@pytest.fixture
def trained_model(sample_data):
    """Returns a fitted model and encoder using the sample dataset."""
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    return model, X, y, encoder, lb


def test_train_model_returns_random_forest(trained_model):
    """train_model should return a fitted RandomForestClassifier."""
    model, _, _, _, _ = trained_model
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_"), "Model should be fitted (has estimators_)"


def test_inference_returns_correct_shape(trained_model):
    """inference should return a 1-D array with one prediction per input row."""
    model, X, _, _, _ = trained_model
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (X.shape[0],)


def test_train_test_split_sizes():
    """An 80/20 train-test split should produce the expected row counts."""
    from sklearn.model_selection import train_test_split

    data = pd.DataFrame({"a": range(100), "b": range(100)})
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    assert len(train) == 80
    assert len(test) == 20
