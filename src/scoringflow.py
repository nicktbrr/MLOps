from metaflow import FlowSpec, Parameter, step
from steps import (
    start_step,
    transform_step,
    dimensionality_reduction_step,
    hyperparameter_tuning_step,
    register_model_step,
    end_step,
    load_model_step,
    evaluate_step
)

class ScoringFlow(FlowSpec):
    """
    A modular scoring flow for regression models.
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
        self.next(self.load_model)
    
    @step
    def load_model(self):
        """Load the best model from MLflow"""
        load_model_step(self)
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        """Evaluate the model"""
        evaluate_step(self)
        self.next(self.end)

    @step
    def end(self):
        """End step"""
        return end_step(self)
    
if __name__ == '__main__':
    ScoringFlow() 
        
    