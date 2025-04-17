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
    dataset = Parameter('boston', help='dataset to use for training', default='../data/boston.csv')
    pca_dimensions = Parameter('pca_dimensions', help='number of PCA dimensions', default=9)

    """
    The start step is the first step in the flow.
    It loads the data and splits it into training and validation sets.
    """
    @step
    def start(self):
        """Starting point - load data"""
        start_step(self)
        self.next(self.transform_data)

    """
    The transform_data step is the second step in the flow.
    It transforms the data with standard scaler.
    """
    @step
    def transform_data(self):
        """Transform and split data"""
        transform_step(self)
        self.next(self.dimensionality_reduction)

    """
    The dimensionality_reduction step is the third step in the flow.
    It applies PCA dimensionality reduction.
    """
    @step
    def dimensionality_reduction(self):
        """Apply PCA dimensionality reduction"""
        dimensionality_reduction_step(self)
        self.next(self.hyperparameter_tuning)

    """
    The hyperparameter_tuning step is the fourth step in the flow.
    It performs hyperparameter tuning.
    """
    @step
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning"""
        hyperparameter_tuning_step(self)
        self.next(self.register_model)

    """
    The register_model step is the fifth step in the flow.
    It registers the best model with MLflow.
    """
    @step
    def register_model(self):
        """Register the best model with MLflow"""
        register_model_step(self)
        self.next(self.end)

    """
    The end step is the last step in the flow.
    It returns the end step.
    """
    @step
    def end(self):
        """End step"""
        return end_step(self)


if __name__ == '__main__':
    TrainingFlow() 