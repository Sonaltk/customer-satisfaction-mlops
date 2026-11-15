import logging
from abc import ABC, abstractmethod

import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Model(ABC):
    @abstractmethod
    def train(self, x_train, y_train, **kwargs):
        pass

    @abstractmethod
    def optimize(self, trial, x_tr, y_tr, x_val, y_val) -> float:
        """Return the validation MSE for the trial's hyperparameters."""
        pass


def _val_mse(model, x_val, y_val) -> float:
    return mean_squared_error(y_val, model.predict(x_val))


class RandomForestModel(Model):
    def train(self, x_train, y_train, **kwargs):
        reg = RandomForestRegressor(random_state=42, n_jobs=-1, **kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_tr, y_tr, x_val, y_val):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 250),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
        return _val_mse(self.train(x_tr, y_tr, **params), x_val, y_val)


class LightGBMModel(Model):
    def train(self, x_train, y_train, **kwargs):
        reg = LGBMRegressor(random_state=42, verbose=-1, **kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_tr, y_tr, x_val, y_val):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        return _val_mse(self.train(x_tr, y_tr, **params), x_val, y_val)


class XGBoostModel(Model):
    def train(self, x_train, y_train, **kwargs):
        reg = xgb.XGBRegressor(random_state=42, **kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_tr, y_tr, x_val, y_val):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
        return _val_mse(self.train(x_tr, y_tr, **params), x_val, y_val)


class LinearRegressionModel(Model):
    def train(self, x_train, y_train, **kwargs):
        reg = LinearRegression(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_tr, y_tr, x_val, y_val):
        return _val_mse(self.train(x_tr, y_tr), x_val, y_val)


class HyperparameterTuner:
    """Tunes on a validation split carved out of the TRAINING data (grouped by
    order), so the test set is never used for model selection."""

    def __init__(self, model, x_train, y_train, groups_train, val_size=0.2, seed=42):
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        tr_idx, val_idx = next(splitter.split(x_train, y_train, groups=groups_train))
        self.model = model
        self.x_tr, self.x_val = x_train.iloc[tr_idx], x_train.iloc[val_idx]
        self.y_tr, self.y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    def optimize(self, n_trials: int = 30):
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(
            lambda t: self.model.optimize(
                t, self.x_tr, self.y_tr, self.x_val, self.y_val
            ),
            n_trials=n_trials,
        )
        return study.best_trial.params, study.best_value