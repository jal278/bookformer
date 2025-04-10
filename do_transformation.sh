#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <vec-clf-file> <parameter>"
    exit 1
fi

VEC_CLF_FILE=$1
PARAMETER=$2

# Loop through files matching the pattern
# NOTE: to actually run on the embeddings files you'd need
# to point this script towards /data/jlehman/embeddings
for FILE in review-sent-embeddings-3s*.pkl; do
    python apply_classifier.py "$FILE" "$VEC_CLF_FILE" "$PARAMETER"
done

for FILE in review-sent-embeddings2-3s*.pkl; do
    python apply_classifier.py "$FILE" "$VEC_CLF_FILE" "$PARAMETER"
done
