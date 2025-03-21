# bookformer

## setup

directories to data need to be set in config.py 

add your openai API key to a new file called _secrets.py (only needed if you want to re-embed reviews)

TODO:
* add script that exports jsonl data if you want to fine-tune a text-based model through e.g. openai API

## to generate some token files (train + val mini sets)

python token_writer.py

## to run demo inference

python demo_inference.py

## to browse sql database

python sql_browser.py

## to embed reviews w/ the openai API

python embed_reviews.py

* note requires installing spacy and downloading the en_core_web_sm model: `python -m spacy download en_core_web_sm`
## to train a logistic regression classifier from pos/negative text examples

python train_classifier.py

## to apply saved classifier across an embedding file

python apply_classifier.py <embedding file> <classifier_file> <output extension>

## pipeline to create a token-based dataset from goodreads reviews + ratings

* first, embed reviews into lots of pickle files that contain dictionaries of (user, book) -> list of sentence embeddings
   * this creates files like review-sent-embeddings-3s-0.pkl, review-sent-embeddings-3s-1.pkl, etc.
* then, run apply_classifier.py across all the pickle files to get a list of (book, score, hash) tuples (do this for each classifier for special tokens)
   * this creates files like review-sent-embeddings-3s-0.pkl.out-wbe.pkl, review-sent-embeddings-3s-0.pkl.out-cml.pkl, etc.
* then, run token_writer.py to create a token-based dataset from goodreads ratings + all of the classifier-output files
* finally, train a model on the token-based dataset (using nanogpt)