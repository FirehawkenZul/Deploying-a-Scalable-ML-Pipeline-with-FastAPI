import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics


def test_one():
    """
    # Used to check and make sure that the model type is Random Forest
    """
    # Create X, y, assign random
    X = np.random.rand(50, 5)
    y = np.random.randint(1, 2, 50)

    # Train model and assign to model
    model = train_model(X, y)
    
    # Final check via assert
    assert isinstance(model, RandomForestClassifier)


def test_two():
    """
    Used to check the inference of the model
    """
    # Create X, y, assign random
    X = np.random.rand(100, 5)
    y = np.random.randint(2, size=100)
    
    # Train model and assign to rand_forest
    rand_forest = train_model(X, y)
    
    # Run inference, assign results to preds
    preds = inference(rand_forest, X)
    
    # Final check via assert
    assert y.shape == preds.shape


def test_three():
    """
    # Used to check computing metrics functions return the expected value
    """
    # Test set creation
    y_true = np.array([0,1,0,1,0])
    y_pred = np.array([0,1,1,1,0])

    # Computing the metrics with compute_model_metrics
    prec, rec, fbeta = compute_model_metrics(y_true, y_pred)
    
    # See if metrics meet values expected
    expec_prec = 0.6667
    expec_rec = 1.0
    expec_fbeta = 0.8
    
    # Final check via assert
    assert round(prec, 4) == expec_prec
    assert round(rec, 4) == expec_rec
    assert round(fbeta, 4) == expec_fbeta
