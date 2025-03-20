import pickle
import numpy as np
import hashlib
import openai
import sklearn
from sklearn.linear_model import LogisticRegression
import sys
import tqdm
from embed_reviews import get_embedding

from config import openai_api_key, data_dir, base_dir


# this script takes in an embedding file, and a classifier file, and applies the classifier to the embeddings
# the output of this script is a list of (book, score, hash) tuples [hash is the hash of the book and user id, which is a unique identifier for the review]
# python apply_classifier.py <embedding_file> <classifier_file> <extension>

#emb file is first argument from command line
emb_file = sys.argv[1]

#classifier file is second argument from command line (a sklearn classifier)
classifier_file = sys.argv[2]

#extension is third argument from command line (this is the extension of the output file, e.g. "wbe" for worst book ever, or "cml" for changed my life)
extension = sys.argv[3]

#output file is the embedding file + ".out" + extension
ofile = emb_file+".out"+extension

print(sys.argv)

# if the classifier file starts with "str:", then we are using an embedding string as an imprompteu classifier (e.g. measuring cosine similarity to a prompt embedding)
# good for prototyping
do_embed = False
if classifier_file[0:4]=='str:':
    do_embed = True
    embed_prompt = classifier_file[4:]

# if the classifier file does not start with "str:", then we are using a pickled sklearn classifier
if not do_embed:
    print("loading classifier...")
    # load in classifier
    _clf = pickle.load(open(classifier_file, 'rb'))
    clf = _clf['classifier']

print("loading embeddings...")
# load in embeddings
_emb = pickle.load(open(emb_file, 'rb'))
done_list = _emb['done']
emb = _emb['emb']

# the output of this script is a list of 
def calc_hash(user,book):
    key = str(book)+" "+str(user)
    hash_val = hashlib.md5(key.encode('utf-8')).hexdigest()
    return hash_val


def cos_sim(a,b):
    a=np.array(a)
    b=np.array(b)
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

sent_embeddings = emb

def score_func(x):
    # if the classifier is a sklearn classifier, then we can use the predict_proba method on a batch of embeddings
    #vec = np.array(x).reshape(1,-1)
    return clf.predict_proba(x)[:,1]

if do_embed:
    embedding_probe = get_embedding(embed_prompt)
    score_func = lambda x: cos_sim(x, embedding_probe)


# for each (user, book) tuple in the embeddings, we can use the score_func to get a score for the review
# we store the max score among the sentences in the review as the score for the review (i.e. we only need one sentence to say the book changed their life to count as a life-changing review)
outs = []
for k, v in tqdm.tqdm(sent_embeddings.items()):
    tmp =[]  
    vecs = []
    for idx,sent in enumerate(v):
        vecs.append(np.array(sent))
    tmp = score_func(vecs)

    # the output of this script is a list of (book, score, hash) tuples
    key = str(k[1])+" "+str(k[0])
    # calculate the hash of the book and user id
    hash_val = hashlib.md5(key.encode('utf-8')).hexdigest()

    # store the max score among the sentences in the review as the score for the review
    outs.append((k[1],max(tmp),hash_val))  # store max closeness among sentences in (book, score, hash) tuples

print("writing out...")
# write out to file
with open(ofile, 'w') as f:
    for k in outs:
        f.write("{}\t{}\t{}\n".format(k[0],k[1],k[2]))