import pandas as pd
import hashlib
import numpy as np
import goodreads
import pickle

greads = goodreads.greads

# load in ratings (user,book,rating,time) sort of pairs
fname = goodreads.base_dir + '/interactions_proc_filtered_new_merged.csv'

print("loading ratings...")
ratings = pd.read_csv(
        fname,
        sep=",",
        usecols=[0, 1, 2,3],
        header=0,
    )
print("sorting ratings...")

# sort by user and time; this is important when we later shuffle users but want to
# keep the time order; we want to be predicting users ratings in order of time so
# we can model dynamic preferences
ratings.sort_values(by=['user','time'], inplace=True)


def csv_to_tokenwork(csv_id):
    """change csv_id of a book into a token-work id (i.e. the numerical token that corresponds to a given 'work')"""
    gread_id = str(greads.convert_book_to_gread_id(csv_id))
    try:
        work = int(greads.convert_book_to_work(gread_id))
    except:
        print(gread_id)
        raise ValueError
    tokenwork = greads.work_to_tokenwork_map[work]
    return tokenwork

csv_book_to_work_map = {}
csv_to_book_map = greads.book_id_map_rev

for key in csv_to_book_map.keys():
    gread_book = greads.convert_book_to_gread_id(key)
    gread_work = greads.convert_book_to_work(str(gread_book))
    try:
        csv_book_to_work_map[key] = int(gread_work)
    except:
        print(gread_book,'work:',gread_work)
        csv_book_to_work_map[key] = 9999999 # saga, book 16148398 (eventually add to gread_work_to_gread_book map also above)

        
ratings['work'] = ratings['item'].map(lambda x: greads.work_to_tokenwork_map[csv_book_to_work_map[x]])

if False:
    # code to make a train/val split
    # split on users; you could also imagine splitting on time (although problematic for late-written books)

    # what is most populous work
    pop_works = ratings['work'].value_counts()

    # find the most populous work
    #most_pop_work = pop_works.idxmax()
    #print(most_pop_work)

    # %%
    # Step 1: Get unique userids
    unique_userids = ratings['user'].unique()

    # shuffle userids
    np.random.shuffle(unique_userids)
    # %%
    # choose validation userids (1%)
    val = 0.0075
    val_userids = unique_userids[:int(len(unique_userids)*val)]
    train_userids = unique_userids[int(len(unique_userids)*val):]
    # %%
    set_val_userids = set(val_userids)
    set_train_userids = set(train_userids)
    # %%
    split = {'val':val_userids,'train':train_userids}
    pickle.dump(split,open(greads.data_dir+'/userid_split.pkl','wb'))

# %%
# load in userid split
split = pickle.load(open(greads.data_dir+'/userid_split.pkl','rb'))
val_userids = split['val']
train_userids = split['train']
set_val_userids = set(val_userids)
set_train_userids = set(train_userids)
unique_userids = ratings['user'].unique()

# %%
def do_shuffle(df):
    random_map = {userid: np.random.rand() for userid in unique_userids}
    df['random_group'] = df['user'].map(random_map)
    df = df.sort_values(by=['random_group', 'time']).drop(columns='random_group')
    return df

df = do_shuffle(ratings)

# TOKEN SCHEME
# 0 = eot
# 1-5 = rating
# 6 + book = books-ids

def calc_hash(user,book):
    """To save memory, we associate a review with a hash of the user and book ids. 
    This function calculates such a hash; so when we are generating tokens, we can
    quickly associate each quantitative review (of a user+book) with a qualitative
    review if it exists (e.g. the text reviews are stored separately and here we
    integrate that information together.
    """
    key = str(book)+" "+str(user)
    hash_val = hashlib.md5(key.encode('utf-8')).hexdigest()
    return hash_val

class token_writer():
    """Class to write out a numpy array of uint16s iteratively
    To have a simple write interface to write token by token to the numpy array, with
    a signal when it's full. Probably could have just used a queue or something, whoops.
    """
    def __init__(self,dataset_size=10000) -> None:
        self.data_set_size = dataset_size
        self.toks = np.zeros(dtype=np.uint16, shape=(self.data_set_size,))
        self.idx = 0
    def write(self,token):
        if self.idx < self.data_set_size:
            self.toks[self.idx] = token
            self.idx += 1
            return True
        else:
            return False
    def full(self):
        return self.idx >= self.data_set_size

idx = 0

def read_in_hashset(extension):
    """A hashset is a set of hashes of user-book pairs that have been associated with a qualitative review token."""
    fname = greads.data_dir + "/token_hashset_"+extension+".pkl"
    return pickle.load(open(fname, 'rb'))

print("reading in hashsets...")
# whatever expressive-review-tokens we want to use among those that have
# been precalculated
hashset_defs = ['cml','gift','assigned','permap','permae','surprise']
hashset_defs += ['weird','best','erotic','wbe','disgust']

# read in all the precomputed hashset data for text reviews
hashset_data = {x:read_in_hashset(x) for x in hashset_defs}


# TOKEN SCHEME
# 0 = eot
# 1-5 = rating
# 6+ = special tokens
# OFFSET+ = book/work tokens
OFFSET = 22
EOT = 0
CML = 6
GIFT = 7
ASSIGN = 8
PERMAP = 9
PERMAE = 10
SURPRISE = 11
WEIRD = 12
BEST = 13 
WORST = 14
DISGUST = 15
EROTIC = 16
NORATE = 17
ACCENT = 18
REVERSE = 19   #if we want to sometimes train on reverse order of ratings (e.g. to anticipate what books a user will need to read to appreciate a target book)
FIM = 20  # fim & fim2 are tokens to try and do a fill-in-the-middle task (e.g. to predict what books most predict a reader changing their tastes to a target book; e.g. what leads a reader to like Atlas Shrugged or something)
FIM2 = 21
# do we want to write numeric ratings? or just have series of books
write_ratings = True
# do we want to write book id or work ids? (work ids seem better -- they map different editions of the same book to the same token)
write_work = True

token_hashsets = [(hashset_data['cml'],CML),(hashset_data['gift'],GIFT),(hashset_data['assigned'],ASSIGN)]
token_hashsets += [(hashset_data['permap'],PERMAP),(hashset_data['permae'],PERMAE),(hashset_data['surprise'],SURPRISE)]
token_hashsets += [(hashset_data['weird'],WEIRD),(hashset_data['best'],BEST),(hashset_data['wbe'],WORST),(hashset_data['disgust'],DISGUST)]
token_hashsets += [(hashset_data['erotic'],EROTIC)]

# actual function to write tokens out
def write_tokens(twrite,df,write_ratings,write_work,token_hashsets,users,idx_start=0,token_dropout=0.0,rating_dropout=0.0,rating_dropout_token=NORATE,user_invert=False,accent_token=ACCENT):
    df_len = len(df)
    cuser = -1
    idx = 0
    while not twrite.full() and idx < df_len:
        row = df.iloc[idx]
        user = row['user']
        
        #skip users not in users (i.e. val users); invert flag means skip users in users
        user_not_present = user not in users
        if (user_not_present and not user_invert) or (user_invert and not user_not_present):
            idx+=1
            if idx%100000==0:
                print(idx,twrite.idx)
            continue

        if user != cuser:
            twrite.write(EOT)
            cuser = user
        
        item = row['item']
        rating = row['label']
        work = row['work']

        # translate user,item to normalized ids
        _user = greads.convert_user_to_gread_id(user)
        _book = greads.convert_book_to_gread_id(item)
        interaction_hash = calc_hash(_user,_book)
        special_token = False

        for hset,tok in token_hashsets:
            if interaction_hash in hset:
                # write accent tokens to indicate the strength of the qualitative review token
                val = hset[interaction_hash]
                if val >= 0.48:
                    twrite.write(accent_token)
                if val >= 0.55:
                    twrite.write(accent_token)
                if val >= 0.62:
                    twrite.write(accent_token)
                if val >= 0.72:
                    twrite.write(accent_token)
                
                
                #write special token
                twrite.write(tok)
                special_token = True

        if not special_token and np.random.rand() >= token_dropout:
            if write_ratings:
                if np.random.rand() >= rating_dropout:
                    twrite.write(rating)
                else:
                    twrite.write(rating_dropout_token)

            if write_work:
                twrite.write(OFFSET+work)
            else:
                twrite.write(OFFSET+item)
        idx+=1
        if idx%100000==0:
            print(idx,twrite.idx)
    print("done...")
    return twrite.idx

# try baby tests
print("writing train tokens...")
twrite = token_writer(dataset_size=1000000)
idx = write_tokens(twrite,df,write_ratings,write_work,token_hashsets,set_train_userids,token_dropout=0.25,rating_dropout=0.1)
twrite.toks = twrite.toks[:twrite.idx]
print(twrite.toks.shape)
#write out token in bin file
twrite.toks.tofile('tokens_train_mini.bin')

print("writing val tokens...")
val_twrite = token_writer(dataset_size=100000)
write_tokens(val_twrite,df,write_ratings,write_work,token_hashsets,set_val_userids)
val_twrite.toks = val_twrite.toks[:val_twrite.idx]
print(val_twrite.toks.shape)
val_twrite.toks.tofile('tokens_val_mini.bin')

"""
# %%
twrite = token_writer(dataset_size=300000000)
# %%
#idx = write_tokens_gatheruser(twrite,df,write_ratings,write_work,token_hashsets,set_train_userids,rating_dropout=0.2,mode='reverse')
idx = write_tokens_gatheruser(twrite,df,write_ratings,write_work,token_hashsets,set_train_userids,rating_dropout=0.2,mode='fim')
#idx = write_tokens_gatheruser(twrite,df,write_ratings,write_work,token_hashsets,set_train_userids,rating_dropout=0.2,mode='fim')

# %% do one epoch of no token-dropout
idx = write_tokens(twrite,df,write_ratings,write_work,token_hashsets,set_train_userids,rating_dropout=0.2)
# %%

# do 3 epochs of token-dropout
idx = write_tokens(twrite,df,write_ratings,write_work,token_hashsets,set_train_userids,token_dropout=0.25,rating_dropout=0.1)
df = do_shuffle(ratings)
idx = write_tokens(twrite,df,write_ratings,write_work,token_hashsets,set_train_userids,token_dropout=0.25,rating_dropout=0.1)
df = do_shuffle(ratings)
idx = write_tokens(twrite,df,write_ratings,write_work,token_hashsets,set_train_userids,token_dropout=0.25,rating_dropout=0.1)
# %%
val_twrite = token_writer(dataset_size=10000000)
write_tokens(val_twrite,df,write_ratings,write_work,token_hashsets,set_val_userids)
# %%

#todo: shorten toks array to be of len idx (if not full)

print(twrite.idx)
print(twrite.toks.shape)
# resize toks array to fit idx size
twrite.toks = twrite.toks[:twrite.idx]
print(twrite.toks.shape)
# %%
#write out token in bin file
twrite.toks.tofile('tokens_v2_1epoch_reg_train.bin')
# %%
twrite.toks = twrite.toks[:twrite.idx]
twrite.toks.tofile("tokens_v2_3epoch_fim_train_2.bin")
# %%
twrite.toks = twrite.toks[:twrite.idx]
twrite.toks.tofile("tokens_v2_1epoch_rev_train.bin")

# %%
val_size = 200000
train_size = twrite.idx - val_size
twrite.toks[:train_size].tofile('tokens_rate_perma1_train.bin')
twrite.toks[train_size:].tofile('tokens_rate_perma1_val.bin')
# %%
val_twrite.toks = val_twrite.toks[:val_twrite.idx]
val_twrite.toks.tofile('tokens_v2_test.bin')
# %%
def mix_bins(binaries,probs,outfile,maxsize,split_token=EOT):
    # mmap all the binaries
    mmaps = [np.memmap(x, dtype=np.uint16, mode='r') for x in binaries]
    idxs = [0]*len(mmaps)
    # open the output file
    out_bin = open(outfile, 'wb')
    written = 0
    _cnt = 0

    while written < maxsize:
        # choose which mmap to read from
        idx = np.random.choice(len(mmaps), p=probs)
        mmap = mmaps[idx]
        _idx = idxs[idx]

        start = _idx
        end = start
        while mmap[end] != split_token or end==0:
            end += 1
        end += 1

        # update the index
        idxs[idx] = end

        # write the chosen slice to the output file
        out_bin.write(mmap[start:end].tobytes())
        written += end - start
        _cnt += 1
        if _cnt % 50000 == 0:
            print(_cnt,written)
# %%
from collections import Counter
# count how many tokens of each type in file
def count_tokens(fname):
    idx = 0
    data = np.memmap(fname, dtype=np.uint16, mode='r')
    counts = Counter()
    for token in data:
        counts[token] += 1
        idx += 1
        if idx % 1000000 == 0:
            print(idx)            
    return counts

tkn_count = count_tokens('tokens_v2_1epoch_reg_train.bin')

# %%
# what are the top 10 tokens
tkn_count.most_common(200)
# what are the least 10 tokens
#tkn_count.most_common()[-50:]

# %%
binaries = ['tokens_v2_1epoch_rev_train.bin','tokens_v2_3epoch_fim_train_2.bin','tokens_v2_3epoch_fim_train.bin','tokens_v2_1epoch_reg_train.bin']
mix_bins(binaries,[0.125,0.375,0.375,0.125],'tokens_v2_train_mixed.bin',250e6)
# %%    
tok_base_dir = '/Users/joel/code/nanoGPT/data/bookgpt/'
binaries = [tok_base_dir+'tokens_rate_perma_tdrop.bin',tok_base_dir+'tokens_rate_perma_train_vanilla.bin']
mix_bins(binaries,[0.5,0.5],tok_base_dir+'tokens_rate_perma_train_mixed.bin',300e6)

# %%


# Assuming df is your DataFrame and 'user_id' is the column of interest
last_occurrences = df.groupby('user').tail(1).index
# Map user_ids to their first iloc index
user_id_to_iloc = {user_id: i for i, user_id in enumerate(df.loc[last_occurrences, 'user'])}

 

# %%
import random
def do_fim(user_rows):
    length = len(user_rows)
    length_min = 3

    cap = 50
    if length>cap:
        beg = random.randint(0,length-cap)
        end = beg+cap
        user_rows = user_rows[beg:end]
        length = len(user_rows)

    if length<length_min:
        return []
    
    beg = random.randint(0,length-length_min)
    end = random.randint(beg+length_min,length)
    user_rows = user_rows[beg:end]
    length = len(user_rows)
    #print(f"making chunk from {beg} to {end}, new length: {length}")

    fim_length = random.randint(1,5)
    fim_length = min(fim_length,len(user_rows)-2)

    #print(fim_length)
    fim_chunk_beg = random.randint(1,length-fim_length-1)
    fim_chunk_end = fim_chunk_beg + fim_length

    chunks = []
    chunks.append(user_rows[:fim_chunk_beg])
    chunks.append(user_rows[fim_chunk_end:])
    chunks.append(user_rows[fim_chunk_beg:fim_chunk_end])
    #print(fim_chunk_beg,fim_chunk_end,len(user_rows),fim_length,len(user_rows))
    #okay now print but with more descriptive text
    #print(f"writing fim chunk of length {fim_length} from {fim_chunk_beg} to {fim_chunk_end} out of {len(user_rows)}")
    return chunks

# %%
def write_tokens_gatheruser(twrite,df,write_ratings,write_work,token_hashsets,users,idx_start=0,token_dropout=0.0,rating_dropout=0.0,rating_dropout_token=NORATE,user_invert=False,accent_token=ACCENT,reverse_token=REVERSE,fim_tokens=[FIM,FIM2],mode='reg'):
    df_len = len(df)
    cuser = -1
    idx = 0
    user_rows = []

    def do_write(user_rows,twrite,mode='reg'):
        if mode=='reg':
            chunks = [user_rows]

        if mode=='reverse':
            #take subset of user_rows and reverse
            length = len(user_rows)
            cap = 50
            if length>cap:
                beg = random.randint(0,length-cap)
                end = beg+cap
                user_rows = user_rows[beg:end]    
                twrite.write(reverse_token)
                user_rows = user_rows[::-1]
            chunks = [user_rows]

        if mode=='fim':
            chunks = do_fim(user_rows)
            if chunks == []:
                return

        for idx,chunk in enumerate(chunks):
            if idx==1 and mode=='fim':
                twrite.write(fim_tokens[0])
            if idx==2 and mode=='fim':
                twrite.write(fim_tokens[1])
            for row in chunk:
                item = row['item']
                rating = row['label']
                work = row['work']
                user = row['user']

                # translate user,item to normalized ids
                _user = greads.convert_user_to_gread_id(user)
                _book = greads.convert_book_to_gread_id(item)

                interaction_hash = calc_hash(_user,_book)
                special_token = False
                for hset,tok in token_hashsets:
                    if interaction_hash in hset:
                        #print(f"[token_writer] {interaction_hash} found in hashset {tok}, {twrite.idx}")
                        val = hset[interaction_hash]
                        if val >= 0.48:
                            twrite.write(accent_token)
                        if val >= 0.55:
                            twrite.write(accent_token)
                        if val >= 0.62:
                            twrite.write(accent_token)
                        if val >= 0.72:
                            twrite.write(accent_token)
                        
                        
                        #write special token
                        twrite.write(tok)

                        special_token = True

                if not special_token and np.random.rand() >= token_dropout:
                    if write_ratings:
                        if np.random.rand() >= rating_dropout:
                            twrite.write(rating)
                        else:
                            twrite.write(rating_dropout_token)

                    if write_work:
                        twrite.write(OFFSET+work)
                    else:
                        twrite.write(OFFSET+item)
        twrite.write(EOT)

    while not twrite.full() and idx < df_len:
        row = df.iloc[idx]
        user = row['user']
        
        #skip users not in users (i.e. val users); invert flag means skip users in users
        user_not_present = user not in users
        if (user_not_present and not user_invert) or (user_invert and not user_not_present):
            idx+=1
            if idx%100000==0:
                print(idx,twrite.idx)
            continue

        if user != cuser:
            do_write(user_rows,twrite,mode=mode)
            cuser = user
            user_rows = []
        
        user_rows.append(row)

        idx+=1
        if idx%100000==0:
            print(idx,twrite.idx)

    print("done...")
    return twrite.idx
# %%
gread_user = '12796e5d0e75c5014b2f212d7e63acf8'
csv_user = greads.convert_user_to_csv_id(gread_user)
user_rows = df[df['user']==csv_user]

# %%
val_twrite1 = token_writer(dataset_size=500)
write_tokens_gatheruser(val_twrite1,df,write_ratings,write_work,token_hashsets,set_train_userids)
val_twrite2 = token_writer(dataset_size=500)
write_tokens_gatheruser(val_twrite2,df,write_ratings,write_work,token_hashsets,set_train_userids,mode='fim')
# %%
import collections
def write_jsonl_specialtoken(df,token_hashsets,users,user_invert=False,max_jsonl=50,threshold=0.0):
    df_len = len(df)
    cuser = -1
    idx = 0
    user_rows = []

    jsonl = []
    special_counts = collections.defaultdict(int)

    def do_find(user_rows):
        chunks = [user_rows]
        special_token_idx = []

        for idx,chunk in enumerate(chunks):
            for _idx,row in enumerate(chunk):
                item = row['item']
                rating = row['label']
                work = row['work']
                user = row['user']

                # translate user,item to normalized ids
                _user = greads.convert_user_to_gread_id(user)
                _book = greads.convert_book_to_gread_id(item)

                interaction_hash = calc_hash(_user,_book)
                special_token = False
                for hset,tok in token_hashsets:
                    if interaction_hash in hset:
                        #print(f"[token_writer] {interaction_hash} found in hashset {tok}, {twrite.idx}")
                        val = hset[interaction_hash]
                        if val >= threshold:
                            special_token = True
                            special_token_idx.append(_idx)
                        else:
                            pass
                            #print(f"skipping {tok} with intensity {val}")

                
                #if write_ratings:
                #twrite.write(rating)
 
                #if write_work:
                #twrite.write(OFFSET+work)
                #else:
                #    twrite.write(OFFSET+item)
        #twrite.write(EOT)
        return special_token_idx
    
    def do_write(user_rows,situation):
        chunks = [user_rows]
        jsonl = []
    
        for idx,chunk in enumerate(chunks):
            for _idx,row in enumerate(chunk):
                item = row['item']
                rating = row['label']
                work = row['work']
                user = row['user']

                # translate user,item to normalized ids
                _user = greads.convert_user_to_gread_id(user)
                _book = greads.convert_book_to_gread_id(item)

                entry = {}
                entry['work'] = work
                entry['item'] = item
                entry['rating'] = rating

                
                interaction_hash = calc_hash(_user,_book)
                special_token = False
                for hset,tok in token_hashsets:
                    if interaction_hash in hset:
                        #print(f"[token_writer] {interaction_hash} found in hashset {tok}, {twrite.idx}")
                        val = hset[interaction_hash]
                        special_token = True
                        entry['special_token'] = tok
                        entry['intensity'] = val

                jsonl.append(entry)
                #if write_ratings:
                #twrite.write(rating)
 
                #if write_work:
                #twrite.write(OFFSET+work)
                #else:
                #    twrite.write(OFFSET+item)
                if _idx == situation:
                    break
        #twrite.write(EOT)
        return jsonl
    
    print(f"writing {max_jsonl} jsonl entries")
    print(f"df_len: {df_len}")
    print(f"users: {len(users)}")
    while idx < df_len and len(jsonl)<max_jsonl:
        row = df.iloc[idx]
        user = row['user']
        
        #skip users not in users (i.e. val users); invert flag means skip users in users
        user_not_present = user not in users
        if (user_not_present and not user_invert) or (user_invert and not user_not_present):
            idx+=1
            if idx%100000==0:
                print(idx,len(jsonl))
            continue

        if user != cuser:
            situations = do_find(user_rows)
            #print(f"found {len(situations)} situations")
            for situation in situations:
                _jsonl = do_write(user_rows,situation=situation)
                jsonl.append(_jsonl)

            cuser = user
            user_rows = []
        
        user_rows.append(row)

        idx+=1
        if idx%100000==0:
            print(idx,len(jsonl))

    print("done...")
    print(f"wrote {len(jsonl)} jsonl entries and {idx} rows")
    return jsonl
# %%
_token_hashsets = [(hashset_data['cml'],CML)]
jsonl = write_jsonl_specialtoken(df,_token_hashsets,set_train_userids,user_invert=False,max_jsonl=3000)
# %%
task_text = {}
task_text[CML] = "it changed my life"
task_text[SURPRISE] = "it surprised me that I liked it"
task_text[WEIRD] = "it was a very weird book"
task_text[EROTIC] = "it was a very erotic book"
task_text[PERMAE] = "it was delightfully challenging"
task_text[BEST] = "it was the best book I've ever read"
task_text[DISGUST] = "it was a disgusting book"
task_text[WORST] = "it was the worst book I've ever read"
task_text[PERMAP] = "it made me happy" 

name_to_token = {}
name_to_token['cml'] = CML
name_to_token['surprise'] = SURPRISE
name_to_token['weird'] = WEIRD
name_to_token['erotic'] = EROTIC
name_to_token['permae'] = PERMAE
name_to_token['best'] = BEST
name_to_token['disgust'] = DISGUST
name_to_token['wbe'] = WORST
name_to_token['permap'] = PERMAP

token_to_name = {v:k for k,v in name_to_token.items()}
# %%
_token_hashsets = [(hashset_data['cml'],CML),(hashset_data['surprise'],SURPRISE),(hashset_data['weird'],WEIRD),(hashset_data['erotic'],EROTIC),(hashset_data['permae'],PERMAE),(hashset_data['best'],BEST),(hashset_data['disgust'],DISGUST),(hashset_data['wbe'],WORST),(hashset_data['permap'],PERMAP)]
jsonl = []
for hset,tok in _token_hashsets:
    print(f"writing {tok}")
    name = token_to_name[tok]
    _jsonl = write_jsonl_specialtoken(df,[(hset,tok)],set_train_userids,user_invert=False,max_jsonl=500,threshold=thresholds[name])
    jsonl.extend(_jsonl)
# %%
random.shuffle(jsonl)
# %%
def format_entry(entry):
    v = entry['work']
    title = tokens_to_titles[v]
    rating = entry['rating']

    out_string = f"{title} -- {rating} stars"
    if 'special_token' in entry:
        if entry['special_token'] == CML:
            out_string += " -- Changed My Life"
        if entry['special_token'] == SURPRISE:
            out_string += " -- Surprised I Liked It"
        if entry['special_token'] == WEIRD:
            out_string += " -- Very Weird Book"
        if entry['special_token'] == EROTIC:
            out_string += " -- Very Sensual Book"
        if entry['special_token'] == PERMAE:
            out_string += " -- Delightfully Challenging"
        if entry['special_token'] == BEST:
            out_string += " -- Best Book I've Ever Read"
        if entry['special_token'] == DISGUST:
            out_string += " -- Disgusting Book"
        if entry['special_token'] == WORST:
            out_string += " -- Worst Book I've Ever Read"
        if entry['special_token'] == PERMAP:
            out_string += " -- Made Me Happy"

    return out_string

def create_json_output(example,max_context=35):
    if len(example)==1:
        return None,None
    example = example[-max_context:]
    input = example[:-1]
    output = example[-1]

    input_string = '\n'.join([format_entry(x) for x in input])
    output_string = format_entry(output)

    return input_string,output_string

def form_into_openai(inp,out,task):
    messages = []
    messages.append({"role": "system", "content": f"You are a helpful assistant that predicts, given some ratings of books they've read, another book meeting a particular criteria. The user will give a list of books and star ratings. You will output the title of another book that the user would say {task}."})
    messages.append({"role": "user", "content": inp})
    messages.append({"role": "assistant", "content": out})
    return {'messages':messages}

inp,out = create_json_output(jsonl[6])
if inp is not None:
    print(inp)
    print('-'*100)
    print(out)
_out = form_into_openai(inp,out,task_text[jsonl[6][-1]['special_token']])
print(_out)

# %%
examples = [create_json_output(x) for x in jsonl]
task_text = [task_text[x[-1]['special_token']] for x in jsonl if x[-1]['special_token'] is not None]
openai_examples = [form_into_openai(x[0],x[1],task_text[i]) for i,x in enumerate(examples) if x[0] is not None]

train_examples = openai_examples[:int(0.85*len(openai_examples))]
val_examples = openai_examples[int(0.85*len(openai_examples)):]

def write_out_jsonl(examples,filename):
    with open(filename,'w') as f:
        for example in examples:
            json.dump(example,f)
            f.write('\n')

write_out_jsonl(train_examples,'all_train_v1.jsonl')
write_out_jsonl(val_examples,'all_val_v1.jsonl')

# %%
# some issue getting titles of works
streamlit_dict = pickle.load(open('streamlit_dict.pkl', 'rb'))
booktitles = streamlit_dict['titles']
titles_to_tokens = streamlit_dict['titles_to_token']
tokens_to_titles = streamlit_dict['token_to_titles']

work_to_tokenwork_map = pickle.load(open(base_dir+'/map_work_to_tokenwork.pkl','rb'))
tokenwork_to_work_map = {v:k for k,v in work_to_tokenwork_map.items()}

# %%
tokenwork_to_work_map[727]
# %%
"""