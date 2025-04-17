from metaflow import step
from sklearn.decomposition import PCA


def dimensionality_reduction_step(self):
    """Apply PCA dimensionality reduction"""
    print("Dimensionality reduction...")
    pca = PCA(n_components=self.pca_dimensions)
    X_train_pca = pca.fit_transform(self.X_train)
    X_val_pca = pca.transform(self.X_val)
    X_test_pca = pca.transform(self.X_test)
    
    self.X_train_pca = X_train_pca
    self.X_val_pca = X_val_pca
    self.X_test_pca = X_test_pca