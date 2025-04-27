from metaflow import step

@step
def end_step(self):
    """End step"""
    print("Training flow completed successfully!") 