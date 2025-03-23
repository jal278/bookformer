from config import embedding_dir, classifier_dir, base_dir
import pickle
import spacy
from embed_reviews import preprocess_sentences
import numpy as np
import gzip
import json
from collections import Counter
from goodreads import greads

"""
Simple script to run a classifier on a subset of review embeddings, then
look through the text reviews to find sentences that match the classifier scores.

A way to get a sense of how the classifier is qualitatively working.
"""

embeddings_file = 'review-sent-embeddings-3s.pkl'
classifier_file = 'vec-clf-m3-cml5.pkl'
verbose = False # if True, print out all hits
top_k = 50 # print out top k sentences
max_hits = 1000 # look for at least this many hits above threshold
thresh = 0.4 # minimal classifier score to consider


#emb_file = 'review-sent-embeddings-{}.pkl'.format(m_abbrev)
emb_file = embedding_dir + "/" + embeddings_file
_emb = pickle.load(open(emb_file, 'rb'))
print("embedding file loaded")
# print the keys of _emb
emb = _emb['emb']
sent_embeddings = emb

print("loading classifier...")
# load in classifier
classifier_file = classifier_dir + "/models/" + classifier_file
_clf = pickle.load(open(classifier_file, 'rb'))
clf = _clf['classifier']

fname = base_dir + '/reviews_filtered.json.gz'
fname2 = base_dir + '/reviews_filtered_2.json.gz'

# embed sentences or whole reviews? empirically sentences work much better, but with better embeddings perhaps could get away with whole reviews
def load_reviews(fname=fname,_reviews=None):
    if _reviews is not None:
        reviews = _reviews
    else:
        reviews = []
    with gzip.open(fname, 'rt') as fin:
        for line in fin:
            reviews.append(json.loads(line))
    return reviews

print("loading reviews...")
reviews = load_reviews(fname)
#reviews = load_reviews(fname2, reviews)


print("applying classifier...")
sims = []
cnt =0
tot = len(sent_embeddings)
vecs = []
for k, v in sent_embeddings.items():
    vecs=[]
    for idx,sent in enumerate(v):
        vec = np.array(sent) #.reshape(1,-1)
        vecs.append(vec)
    _sims = clf.predict_proba(vecs)[:,1]
    for idx,sent in enumerate(v):
        sims.append((k, _sims[idx],idx))
    #sims.append((k, clf.predict_proba(vec)[0,1],idx))
    if cnt%1000==0:
        print(cnt, tot)
    cnt+=1

print("done")
sims = sorted(sims, key=lambda x: x[1], reverse=True)

# could visualize the distribution of scores here

# could examing a particular book if we wanted to
b_id = '10818853'

book_counter = Counter()
user_counter = Counter()

# let's set minimal threshold here 
idx = 0
entry = sims[idx]
entries = {}
while entry[1] > thresh:
    rev = entry[0]
    score = entry[1]
    (user,book) = rev
    book_counter[book]+=1
    user_counter[user]+=1
    if True: #book == b_id:
        entries[(user,book)] = entry
    idx+=1
    entry = sims[idx]
# %%

print("Looking through text reviews to correlate with classifier scores...")
book_counter_rev = Counter()
user_counter_rev = Counter()
scored_sentences = []
cnt = 0
hits = 0
for review in reviews[:100000]:
    book_counter_rev[review['book_id']] += 1
    user_counter_rev[review['user_id']] += 1
    if (review['user_id'], review['book_id']) in entries:
        hits+=1
        if verbose:
            print(review['review_text'])
        sentences = preprocess_sentences(review['review_text'])
        # we know this review is included in entries; now to find all sentences from entries that apply here
        entry = entries[(review['user_id'], review['book_id'])]
        sent = entry[-1]
        score = entry[1]
        #print(sent,score,review['book_id'])
        print(score,sentences[sent],greads.titles[review['book_id']],review['user_id'])
        if verbose:
            print("--sep---")
        scored_sentences.append((score,sentences[sent],greads.titles[review['book_id']]))

    cnt += 1
    if cnt % 50000 == 0:
        print(cnt)
    if hits >= max_hits:
        break
# %%

# sort by score & print top k
scored_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)
for score,sent,title in scored_sentences[:top_k]:
    print(score,sent,"[[",title,"]]")
