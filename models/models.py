import logging

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import FeatureUnion, Pipeline

from configs import options


class Model:
    def __init__(self, pipeline, unions, target_transformer=None):
        self.steps = self._create_steps(pipeline, unions)
        self.target_transformer = self._make_step(target_transformer)
        self.model = None

    def create(self):
        self.model = Pipeline(steps=self.steps)
        # if self.target_transformer is not None:
        #     self.model = TransformedTargetRegressor(
        #         regressor=self.model, transformer=self.target_transformer[1]
        #     )
        return self.model

    def _create_steps(self, pipeline, unions):
        steps = list()
        for model_name in pipeline:
            # add features from pipeline
            if model_name in options.METHODS.keys():
                step = self._make_step(model_name)
                steps.append(step)
            # add combined features

            elif model_name in unions.keys():
                steps_cf = list()
                for model_name_cf in unions[model_name]:
                    if model_name_cf in options.METHODS.keys():
                        step = self._make_step(model_name_cf)
                        steps_cf.append(step)
                if steps_cf:
                    steps.append([model_name, FeatureUnion(steps_cf)])

            else:
                logging.warning(f"Model {model_name} not found in options.METHODS.keys()")
                steps.append([model_name, None])

        return steps

    def _make_step(self, model_name):
        if model_name is None:
            return None

        if isinstance(options.METHODS[model_name], type):
            step = [model_name, options.METHODS[model_name]()]
        else:
            # if already initialized
            step = [model_name, options.METHODS[model_name]]
        return step
