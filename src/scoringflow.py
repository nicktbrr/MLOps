from metaflow import FlowSpec, Parameter, step, conda_base, resources, retry
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

    dataset = Parameter('boston', help='dataset to use for training', default='../data/boston.csv')
    pca_dimensions = Parameter('pca_dimensions', help='number of PCA dimensions', default=9)

    """
    The start step is the first step in the flow.
    It loads the data and splits it into training and validation sets 
    and importantly for us the test set.
    """
    @step
    @retry(times=2)
    @resources(memory=1000)
    @conda_base(
    libraries={
        'numpy': '1.23.5',
        'pandas': '1.4.4',
        'scikit-learn': '1.2.2',
        'mlflow': '2.5.0',
        "databricks-cli": "0.17.6",  # or latest stable version
        "google-cloud-secret-manager": "2.7.0",
        "hyperopt": "0.2.7",
    },
    python='3.10.9'
    )
    def start(self):
        """Starting point - load data"""
        start_step(self)
        self.next(self.transform_data)

    """
    The transform_data step is the second step in the flow.
    It transforms the data with standard scaler.
    """
    @step
    @retry(times=2)
    @resources(memory=1000)
    @conda_base(
    libraries={
        'numpy': '1.23.5',
        'pandas': '1.4.4',
        'scikit-learn': '1.2.2',
        'mlflow': '2.5.0',
        "databricks-cli": "0.17.6",  # or latest stable version
        "google-cloud-secret-manager": "2.7.0",
        "hyperopt": "0.2.7",
    },
    python='3.10.9'
    )
    def transform_data(self):
        """Transform and split data"""
        transform_step(self)
        self.next(self.dimensionality_reduction)

    """
    The dimensionality_reduction step is the third step in the flow.
    It applies PCA dimensionality reduction.
    """
    @step
    @retry(times=2)
    @resources(memory=1000)
    @conda_base(
    libraries={
        'numpy': '1.23.5',
        'pandas': '1.4.4',
        'scikit-learn': '1.2.2',
        'mlflow': '2.5.0',
        "databricks-cli": "0.17.6",  # or latest stable version
        "google-cloud-secret-manager": "2.7.0",
        "hyperopt": "0.2.7"
    },
    python='3.10.9'
    )
    def dimensionality_reduction(self):
        """Apply PCA dimensionality reduction"""
        dimensionality_reduction_step(self)
        self.next(self.load_model)

    """
    The load_model step is the fourth step in the flow.
    It loads the best registered model from MLflow.
    """
    @step
    @retry(times=2)
    @resources(memory=1000)
    @conda_base(
    libraries={
        'numpy': '1.23.5',
        'pandas': '1.4.4',
        'scikit-learn': '1.2.2',
        'mlflow': '2.5.0',
        "databricks-cli": "0.17.6",  # or latest stable version
        "google-cloud-secret-manager": "2.7.0", 
        "hyperopt": "0.2.7"
    },
    python='3.10.9'
    )
    def load_model(self):
        """Load the best model from MLflow"""
        load_model_step(self)
        self.next(self.evaluate_model)

    """
    The evaluate_model step is the fifth step in the flow.
    It evaluates the model on the test set.
    """
    @step
    @retry(times=2)
    @resources(memory=1000)
    @conda_base(
    libraries={
        'numpy': '1.23.5',
        'pandas': '1.4.4',
        'scikit-learn': '1.2.2',
        'mlflow': '2.5.0',
        "databricks-cli": "0.17.6",  # or latest stable version
        "google-cloud-secret-manager": "2.7.0",
        "hyperopt": "0.2.7"
    },
    python='3.10.9'
    )
    def evaluate_model(self):
        """Evaluate the model"""
        evaluate_step(self)
        self.next(self.end)

    """
    The end step is the last step in the flow.
    It returns the end step.
    """
    @step
    @retry(times=2)
    @resources(memory=1000)
    @conda_base(
    libraries={
        'numpy': '1.23.5',
        'pandas': '1.4.4',
        'scikit-learn': '1.2.2',
        'mlflow': '2.5.0',
        "databricks-cli": "0.17.6",  # or latest stable version
        "google-cloud-secret-manager": "2.7.0",
        "hyperopt": "0.2.7"
    },
    python='3.10.9'
    )
    def end(self):
        """End step"""
        return end_step(self)
    
if __name__ == '__main__':
    ScoringFlow() 
        
    