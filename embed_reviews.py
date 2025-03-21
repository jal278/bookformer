# Ultimate output we want:
# - associate a book,user tuple with [review/did_not_review] [transformation/no_transformation]

# code to write out reviews in jsonl format we then read in here...
#ofile = base_dir + '/reviews_filtered.json.gz'
#with gzip.open(ofile, 'wt') as fout:
#    for d in reviews:
#        fout.write(json.dumps(d) + '\n')

import gzip
import json
import pandas as pd
import openai
import tqdm
import pickle
import numpy
import numpy as np
from numpy import dot
from numpy.linalg import norm
import spacy
from collections import defaultdict
import hashlib
import openai
import time

from config import data_dir, base_dir, openai_api_key
api_key = openai_api_key

fname = base_dir + '/reviews_filtered.json.gz'
fname2 = base_dir + '/reviews_filtered_2.json.gz'

# embed sentences or whole reviews? empirically sentences work much better, but with better embeddings perhaps could get away with whole reviews
mode = "sentence"
print("loading spacy model")
nlp = spacy.load('en_core_web_sm') # Load the English Model
print("spacy model loaded")

def load_reviews(fname=fname,_reviews=None):
    if _reviews is not None:
        reviews = _reviews
    else:
        reviews = []
    with gzip.open(fname, 'rt') as fin:
        for line in fin:
            reviews.append(json.loads(line))
    return reviews

reviews = load_reviews(fname)
reviews = load_reviews(fname2, reviews)


client = openai.OpenAI(api_key=api_key)
default_embedding_model = "text-embedding-3-small"

# helper functions
def get_embedding(text, model=default_embedding_model):
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# batching is needed for efficiency (fewer calls to openai)
def get_embedding_batch(text, model = default_embedding_model):
   return client.embeddings.create(input = text, model=model)

#ret = get_embedding_batch(["Best book ever", "Worst book ever","Blah blah blah"])

def preprocess(review):
    """ Simple preprocessing of the review text. Get rid of next-lines, remove direct book quotes (to avoid false-positives where we confuse text 
    *from* the book with text *about* the book. E.g. if a line in the book talks about a character changing *their* life."""
    review = review.replace("\n", " ")
    # remove quoted text from review
    review = remove_quotes(review)

    # TODO: might get more accuracy if we don't do this truncation; we take first 1000 and last 1000 characters if it is a very long review.
    if len(review) > 2000:
        review = review[:1000] + '...' + review[-1000:]

    return review

#create a function that takes a text string, which may have multiple quoted passages, and remove all the quoted passages
def remove_quotes(text):
    # find all double quotes
    quotes_idx = [i for i, ltr in enumerate(text) if ltr == '"']
    # if there are an odd number of quotes, then we have a problem
    if len(quotes_idx) % 2 == 1:
        print("odd number of quotes")
        return text
    # if there are no quotes, then we are done
    if len(quotes_idx) == 0:
        return text
    # otherwise, remove the quoted text
    new_text = ''
    idx = 0
    while idx < len(text):
        if idx in quotes_idx:
            # skip to the next quote
            idx = quotes_idx[quotes_idx.index(idx)+1]
            new_text += '"[quote]."'     # TODO: think about if replacing a quote with this is bad/good
        else:
            new_text += text[idx]
        idx += 1

    # remove extra white space (double spaces)
    new_text = new_text.replace('  ', ' ')
    new_text = new_text.strip()
    return new_text

def save_embs():
    # write embeddings out to disk
    pickle.dump(embeddings, open('review-embeddings.pkl', 'wb'))

def load_embs():
    embeddings = pickle.load(open('review-embeddings.pkl', 'rb'))
    return embeddings


def cos_sim(a,b):
    """ Calculate the cosine similarity between two vectors. """
    a=np.array(a)
    b=np.array(b)
    return dot(a, b)/(norm(a)*norm(b))

def calc_sims(embeddings, embedding_probe):
    """ Calculate the cosine similarity between the probe embedding and all the embeddings in the dictionary. 
    Returns a list of sorted items by similarity."""
    sims = []
    for k, v in embeddings.items():
        sims.append((k, cos_sim(v, embedding_probe)))

    # sort by similarity
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims

def preprocess_sentences(review,total_char_size=3000):
    review = review.replace("\n", " ")
    # remove quoted text from review
    review = remove_quotes(review)

    # use spacy to split the review into sentences 
    # TODO: perhaps some benefit from using a better sentence splitting model?
    doc = nlp(review)
    _sentences = [str(sent).strip() for sent in doc.sents]

    # if the review is too long, clip it
    if len(review) > total_char_size:
        sentences = []
        clip_size = total_char_size // 2

        # first go from beginning finding all sentences until we reach clip_size
        size_so_far = 0
        for idx, sent in enumerate(_sentences):
            size_so_far += len(sent)
            if size_so_far > clip_size:
                break
            sentences.append(sent)

        rev_sentences = []
        # now go from end finding all sentences until we reach clip_size
        size_so_far = 0
        for idx, sent in enumerate(reversed(_sentences)):
            size_so_far += len(sent)
            if size_so_far > clip_size:
                break
            rev_sentences.append(sent)
        
        rev_sentences.reverse()
        sentences.extend(rev_sentences)
        return sentences
    else:
        # if the review is not too long, return the sentences as is
        return _sentences
    


# do sentence-level embeddings across some subset of reviews; we'll call this script many times with different subsets to cover all reviews
if __name__=='__main__' and mode=='sentence':
    done_list = set()
    sent_embeddings = defaultdict(list)

    model = "text-embedding-3-small"
    m_abbrev = "3s"
    batch_size = 300

    # TODO: make this a command line argument; right now you manually have to change this as you
    # iterate through the reviews; it takes several days to embed all reviews (or at least it did last time)
    num = 0   # 0 for first X reviews, 1 for next X reviews, etc.

    # filename for storing embeddings in pkl file
    emb_file = 'review-sent-embeddings-{}-{}.pkl'.format(m_abbrev,num)

    # we want to maintain a list of reviews that we have already processed
    done_file = 'review-sent-embeddings-done-{}.pkl'.format(m_abbrev)

    print("loading reviews")
    reviews = load_reviews()

    # load the done list from disk (done list is a set of (user, book) tuples indexing reviews that have already been processed)
    try:
        _done = pickle.load(open(done_file, 'rb'))
        done_list = _done['done']
    except:
        print("no done file")

    print("len(done_list) = {}".format(len(done_list)))

    idx=0
    stack = []
    for review in tqdm.tqdm(reviews[:]):
        if len(sent_embeddings)>275000:
            break
        passed=False
        review_text = review['review_text']
        if len(review_text.strip()) ==0:
            idx+=1
            # skip whitespace-only reviews
            continue

        # get the user and book id of the review
        user = review['user_id']
        book = review['book_id']

        # skip reviews that have already been processed
        if (user, book) not in done_list:
            # process the review into sentences
            sentences = preprocess_sentences(review['review_text'])

            # embed each sentence
            for sent in sentences:
                # skip whitespace-only sentences
                if len(sent.strip()) ==0:
                    continue
                stack.append( (user, book, sent))
                
                # run batch of embeddings (300 at a time; 300 is arbitrary)
                if len(stack) >= batch_size:
                    print("batch running...")
                    try:
                        ret = get_embedding_batch([k[2] for k in stack], model=model)
                        print(len(sent_embeddings))
                        time.sleep(1)
                    except Exception as e:
                        print(stack)
                        print(e)
                        print([k[2] for k in stack])
                        raise e
                    
                    # now add the results of the batch of embeddings to the sent_embeddings dictionary
                    for i in range(len(stack)):
                        sent_embeddings[(stack[i][0], stack[i][1])].append(np.array(ret.data[i].embedding, dtype=np.float32))
                    stack = []
            # we've processed all sentences in the review, so add the (user, book) tuple to the done list
            done_list.add((user, book))
        else:
            # skip reviews that have already been processed
            passed=True
            
        # save the done list and embeddings every 20000 reviews in case of crash
        idx+=1
        if idx % 20000 == 0 and not passed:
            pickle.dump({'done':done_list}, open(done_file, 'wb'))
            pickle.dump({'done':done_list,'emb':sent_embeddings}, open(emb_file, 'wb'))

    # save the done list and embeddings
    pickle.dump({'done':done_list}, open(done_file, 'wb'))
    pickle.dump({'done':done_list,'emb':sent_embeddings}, open(emb_file, 'wb'))

    # the done list is only important as a way to embed each review once; it is not used after the process is done
    # but of course all the embedding files are important



"""
if __name__=='__main__' and mode=='whole-review':
    idx=0
    stack = []
    batch=True
    for review in tqdm.tqdm(reviews[:5000]):
        review_text = preprocess(review['review_text'])
        user = review['user_id']
        book = review['book_id']
        if (user, book) not in embeddings and len(review_text) > 0:
            #print(review_text)
            if batch:
                stack.append( (user, book, review_text))
                if len(stack) == 100:
                    #print("batch")
                    ret = get_embedding_batch([k[2] for k in stack], model='text-embedding-ada-002')
                    for i in range(len(stack)):
                        embeddings[(stack[i][0], stack[i][1])] = ret.data[i].embedding
                    stack = []
            else:
                embeddings[(user, book)] = get_embedding(review_text, model='text-embedding-ada-002')
        else:
            pass #print("hit")
        idx+=1
        if idx % 10000 == 0:
            pickle.dump(embeddings, open('review-embeddings.pkl', 'wb'))
"""