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

## to train a logistic regression classifier from pos/negative text examples

python train_classifier.py

## to apply saved classifier across an embedding file

python apply_classifier.py <embedding file> <classifier_file> <output extension>
