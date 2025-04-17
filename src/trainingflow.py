from metaflow import FlowSpec, Parameter, step
from steps import (
    start_step,
    transform_step,
    dimensionality_reduction_step,
    hyperparameter_tuning_step,
    register_model_step,
    end_step
)

class TrainingFlow(FlowSpec):
    """
    A modular training flow for regression models.
    Each step is defined in a separate file for better reusability.
    """
    dataset = Parameter('boston', help='dataset to use for training', default='../data/boston.csv')
    pca_dimensions = Parameter('pca_dimensions', help='number of PCA dimensions', default=9)

    @step
    def start(self):
        """Starting point - load data"""
        start_step(self)
        self.next(self.transform_data)

    @step
    def transform_data(self):
        """Transform and split data"""
        transform_step(self)
        self.next(self.dimensionality_reduction)

    @step
    def dimensionality_reduction(self):
        """Apply PCA dimensionality reduction"""
        dimensionality_reduction_step(self)
        self.next(self.hyperparameter_tuning)

    @step
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning"""
        hyperparameter_tuning_step(self)
        self.next(self.register_model)

    @step
    def register_model(self):
        """Register the best model with MLflow"""
        register_model_step(self)
        self.next(self.end)

    @step
    def end(self):
        """End step"""
        return end_step(self)


if __name__ == '__main__':
    TrainingFlow() 