# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(50)

# Create a KMeans instance: kmeans
kmeans = KMeans(6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)
