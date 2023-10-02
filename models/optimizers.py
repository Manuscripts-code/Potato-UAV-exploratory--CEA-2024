import logging
from datetime import datetime

import mlflow
import optuna
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator, cross_val_score
from sklearn.pipeline import Pipeline

from configs import configs
from configs.parser import OptimizerConfig
from data_structures.schemas import StructuredData
from database import schemas  # noqa: F401 - needs to be imported for SQLModel to create tables


class Optimizer:
    def __init__(
        self,
        data_train: StructuredData,
        data_val: StructuredData,
        model: Pipeline,
        validator: BaseCrossValidator,
        optimizer_cfg: OptimizerConfig,
    ):
        self.data_train = data_train
        self.data_val = data_val  # currently unused
        self.model = model
        self.validator = validator
        self.optimizer_cfg = optimizer_cfg

        self.n_trials = optimizer_cfg.n_trials
        self.timeout = optimizer_cfg.timeout
        self.n_jobs = optimizer_cfg.n_jobs
        self.scoring_metric = optimizer_cfg.scoring_metric
        self.scoring_mode = optimizer_cfg.scoring_mode
        self.tuned_params = optimizer_cfg.tuned_parameters

        self._best_model = None
        self._best_trial = None

        # self._run_id = mlflow.active_run().info.run_id

    def run(self):
        study = self._perform_search()
        self._best_trial = study.best_trial
        self._refit_model(self._best_trial.params)
        logging.info(f"Best {self.scoring_metric}: {self._best_trial.value}")
        logging.info(f"Best hyperparameters found were: {self._best_trial.params}")

    def _perform_search(self):
        study = optuna.create_study(
            direction=self.scoring_mode,
            storage=f"sqlite:///{configs.DB_PATH}",
            study_name=f"trial--{datetime.now().strftime(configs.DATETIME_FORMAT)}",
            # load_if_exists=True,
        )
        study.optimize(self._trainable, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs)
        return study

    def _trainable(self, trial):
        params = self._trial_params(trial)
        score = self._objective(params)
        return score

    def _trial_params(self, trial):
        params = {}
        for method, values in self.tuned_params.optimize_int.items():
            params[method] = trial.suggest_int(method, *values, log=True)

        for method, values in self.tuned_params.optimize_float.items():
            params[method] = trial.suggest_float(method, *values, log=True)

        for method, values in self.tuned_params.optimize_category.items():
            params[method] = trial.suggest_categorical(method, values)

        return params

    def _objective(self, params):
        self.model = clone(self.model)
        self.model.set_params(**params)
        return self._scorer(self.model)

    def _scorer(self, model):
        score = cross_val_score(
            model,
            X=self.data_train.data,
            y=self.data_train.target.value,
            groups=None,
            scoring=self.scoring_metric,
            cv=self.validator,
            n_jobs=1,
            verbose=0,
            fit_params=None,
            pre_dispatch=1,
            error_score=0,
        )
        return score.mean()

    def _refit_model(self, best_params):
        mlflow.sklearn.autolog()
        self._best_model = clone(self.model)
        self._best_model.set_params(**best_params)
        # Note: data needs to be numpy array for zenml server to work..
        self._best_model.fit(
            self.data_train.data.to_numpy(),
            self.data_train.target.value.to_numpy(),
        )

    @property
    def best_model(self):
        return self._best_model

    @property
    def best_trial(self):
        return self._best_trial
