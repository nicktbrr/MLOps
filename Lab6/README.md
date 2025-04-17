# Modular Training Flow

This project implements a modular training flow for regression models using Metaflow. The flow is split into separate components for better reusability and maintainability.

## Project Structure

```
Lab6/
├── training_flow.py       # Main flow file that orchestrates all steps
├── steps/                 # Directory containing individual step implementations
│   ├── __init__.py        # Package initialization
│   ├── start_step.py      # Data loading step
│   ├── transform_step.py  # Data transformation step
│   ├── dimensionality_reduction_step.py  # PCA step
│   ├── hyperparameter_tuning_step.py     # Model tuning step
│   ├── register_model_step.py            # Model registration step
│   └── end_step.py        # End step
└── mlflow.db              # MLflow tracking database
```

## Usage

To run the training flow:

```bash
python -m Lab6.training_flow
```

## Benefits of Modular Structure

1. **Reusability**: Each step can be reused in different flows
2. **Maintainability**: Easier to update individual components
3. **Testability**: Each component can be tested independently
4. **Readability**: Clearer code organization
5. **Collaboration**: Different team members can work on different components

## Flow Steps

1. **Start**: Load data from CSV file
2. **Transform Data**: Split data into train/validation/test sets and apply scaling
3. **Dimensionality Reduction**: Apply PCA to reduce feature dimensions
4. **Hyperparameter Tuning**: Tune model hyperparameters using Hyperopt
5. **Register Model**: Register the best model with MLflow
6. **End**: Complete the flow 