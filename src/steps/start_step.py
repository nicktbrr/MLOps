from metaflow import step
import pandas as pd


def start_step(self):
    """Starting point - load data"""
    print("Loading data...")
    self.data = pd.read_csv(self.dataset) 