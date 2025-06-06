from metaflow import FlowSpec, Parameter, step, conda_base, resources, retry
from steps import (
    start_step,
    transform_step,
    dimensionality_reduction_step,
    hyperparameter_tuning_step,
    register_model_step,
    end_step
)

@conda_base(
libraries={
    'numpy': '1.23.5',
    'pandas': '2.0.0',
    'scikit-learn': '1.2.2',
    'mlflow': '2.5.0',
    "databricks-cli": "0.17.6",  # or latest stable version
    "google-cloud-secret-manager": "2.7.0",
    'google-cloud-storage': '2.7.0',
    "hyperopt": "0.2.7",
},
python='3.11.11'
)
class TrainingFlow(FlowSpec):
    dataset = Parameter('boston', help='dataset to use for training', default='../data/boston.csv')
    pca_dimensions = Parameter('pca_dimensions', help='number of PCA dimensions', default=9)

    """
    The start step is the first step in the flow.
    It loads the data and splits it into training and validation sets.
    """
    @retry(times=2)
    @step
    def start(self):
        """Starting point - load data"""
        start_step(self)
        self.next(self.transform_data)

    """
    The transform_data step is the second step in the flow.
    It transforms the data with standard scaler.
    """
    @retry(times=2)
    @resources(memory=1000)
    @step
    def transform_data(self):
        """Transform and split data"""
        transform_step(self)
        self.next(self.dimensionality_reduction)

    """
    The dimensionality_reduction step is the third step in the flow.
    It applies PCA dimensionality reduction.
    """
    @retry(times=2)
    @resources(memory=1000)
    @step
    def dimensionality_reduction(self):
        """Apply PCA dimensionality reduction"""
        dimensionality_reduction_step(self)
        self.next(self.hyperparameter_tuning)

    """
    The hyperparameter_tuning step is the fourth step in the flow.
    It performs hyperparameter tuning.
    """
    
    @retry(times=2)
    @resources(memory=1000)
    @step
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning"""
        hyperparameter_tuning_step(self)
        self.next(self.register_model)

    """
    The register_model step is the fifth step in the flow.
    It registers the best model with MLflow.
    """
    
    @retry(times=2)
    @resources(memory=1000)
    @step
    def register_model(self):
        """Register the best model with MLflow"""
        register_model_step(self)
        self.next(self.end)

    """
    The end step is the last step in the flow.
    It returns the end step.
    """
    
    @retry(times=2)
    @resources(memory=1000)
    @step
    def end(self):
        """End step"""
        return end_step(self)


if __name__ == '__main__':
    TrainingFlow() 