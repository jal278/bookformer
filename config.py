# Configuration settings for the bookformer project

# Directory paths
base_dir = '/Users/joel/Downloads/reviews'
data_dir = '/Users/joel/code/examples' 
model_dir = '/Users/joel/code/bookformer'
embedding_dir = '/Users/joel/code/examples/embeddings'
classifier_dir = '/Users/joel/code/examples/classifiers'

try:
    from _secrets import openai_api_key
except ImportError:
    print("No secrets file found, no openai key will be set")
    openai_api_key = None
