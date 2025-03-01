base_dir = '/Users/joel/Downloads/reviews'
data_dir = '/Users/joel/code/examples'

import pickle
import gzip
import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
import json
import tqdm

class goodreads_analysis:
    """
    This class is used to analyze the goodreads data.

    Some weirdness in how books are referenced in the data; goodreads assigns a book id to each edition/version of a book (called 'gread_id'). But then there is also a 'work_id' that applies to all editions/versions of a book.
    The 'work_id' is what we use to map to the tokenized data, to a 'token-work' id. There is also a book id that is renumbered to be sequential (called 'csv id') which saves memory.
    This class contains methods to convert between these ids.

    There are two types of user_ids as well; one is the greads user id, which is the id that goodreads assigns to each user. The other is the csv user id, which is the id that is renumbered to be sequential (called 'csv id') which saves memory.

    The class also contains methods to filter the data, and pull in various subviews of the data (like author names, book titles, etc.)
    """
    def __init__(self,base_dir=base_dir,data_dir=data_dir):
        self.book_dict = {}
        self.user_dict = {}
        self.book_id_map = {}
        self.user_id_map = {}
        self.book_id_map_rev = {}
        self.user_id_map_rev = {}
        self.title_dict = {}
        self.target_books = {}
        self.base_dir = base_dir
        self.data_dir = data_dir
        # load in the book titles
        a= open(self.base_dir + "/book_title_dict.pkl","rb")
        self.titles = pickle.load(a)

        # load in the book id map (map from csv id to book id)
        self.book_id_map = pickle.load(open(base_dir + '/book_id_map_f.pkl', 'rb'))   
        # load in the user id map (map from csv id to user id)
        self.user_id_map = pickle.load(open(base_dir + '/user_id_map_f2.pkl', 'rb'))

        # create a reverse map for book ids
        self.book_id_map_rev = {v:k for k,v in self.book_id_map.items()}
        self.user_id_map_rev = {v:k for k,v in self.user_id_map.items()}

        # load in the book to work map (map from book id to work id)
        self.book_work_map = pickle.load(open(base_dir +'/map_book_to_work.pkl','rb'))
        # load in the work to book map (map from work id to book id)
        self.work_book_map = pickle.load(open(base_dir +'/map_work_to_book.pkl','rb'))

        # load in the work to token-work map (map from work id to token work id)
        self.work_to_tokenwork_map = pickle.load(open(base_dir+'/map_work_to_tokenwork.pkl','rb'))
        # create a reverse map for token work ids
        self.tokenwork_to_work_map = {v:k for k,v in self.work_to_tokenwork_map.items()}

    def convert_tokenwork_to_book(self,tokenwork):
        work = self.tokenwork_to_work_map[tokenwork]
        work = str(work)
        return self.convert_work_to_book(work)

    def convert_work_to_tokenwork(self,work):
        return self.work_to_tokenwork_map[work]
    
    def convert_work_to_book(self,work):
        return self.work_book_map[work]
    
    def convert_book_to_work(self,book):
        return self.book_work_map[str(book)]

    def convert_book_to_csv_id(self,book_id):
        return self.book_id_map[book_id]

    def convert_book_to_gread_id(self,book_csv_id):
        return self.book_id_map_rev[book_csv_id]
    
    def convert_user_to_csv_id(self,user_csv_id):
        return  self.user_id_map[user_csv_id]
    
    def convert_user_to_gread_id(self,user_id):
        return self.user_id_map_rev[user_id]

    def get_best_books(self,threshold=3):
        """
        Grab books with at least *threshold* number of reviews 
        """
        file_name = self.base_dir + '/goodreads_books.json.gz'
        print('counting file:', file_name)
        n_review = 0
        print('current line: ', end='')
        data = []
        with gzip.open(file_name) as fin:
            for l in fin:
                d = json.loads(l)
                if d['book_id'] in self.target_books:
                    num = self.target_books[d['book_id']]
                    if num>threshold:
                        print("found...", d["title"], num)
                    data.append(d)
                n_review += 1

                if n_review % 100000 == 0:
                    print(n_review, end=',')
        return data
    
    def count_reviews(self):
        '''
        Calculate number of text reviews [but also this is directly linked in the books data in the field 'text_reviews_count']
        '''
        file_name = self.base_dir + '/goodreads_reviews_dedup.json.gz'
        review_count = Counter()
        print('counting file:', file_name)
        n_review = 0
        print('current line: ', end='')
        with gzip.open(file_name) as fin:
            for l in fin:
                d = json.loads(l)
                if n_review % 1000000 == 0:
                    print(n_review, end=',')
                n_review += 1
                review_count[d['book_id']] += 1
        print('complete')
        print('done!')
        return n_review, review_count
    
    def filter_reviews(self,uc_keep_ids=None):
        """
        process reviews file and filter out reviews that are not in the filtered books or filtered users
        """
        file_name = self.base_dir + '/goodreads_reviews_dedup.json.gz'

        #read in keep_ids (created through another script)
        with open('bc_keep_ids.pkl', 'rb') as f:
            bc_keep_ids = pickle.load(f)
            
        if uc_keep_ids is None:        
            with open('uc_keep_ids.pkl', 'rb') as f:
                uc_keep_ids = pickle.load(f)

        print('current line: ', end='')

        reviews = []
        tot_reviews = 0
        n_review = 0
        with gzip.open(file_name) as fin:
            for l in fin:
                d = json.loads(l)
                if n_review % 100000==0:
                    print(n_review, tot_reviews)

                n_review += 1
                if int(d['book_id']) in bc_keep_ids and d['user_id'] in uc_keep_ids:
                    reviews.append(d)
                    tot_reviews+=1
        return reviews

    def keyphrase_reviews_in_book(self,phrase='changed my life',review_threshold=40,hit_threshold=3,do_reviews=True,verbose=False):
        """
        Find reviews that contain the phrase 'phrase', from books with at least *review_threshold* reviews, and at least *hit_threshold* reviews that contain the phrase to be included in ranked results.
        Motivation is to find books that contain 'phrase' more often than chance in a statistically meaningful way, which is obscured if the sample size is small of either total number of reviews or number of reviews that contain the phrase.
        """
        file_name = self.base_dir + '/goodreads_reviews_dedup.json.gz'

        phrase = phrase.upper()
        book_count = Counter()
        review_count = Counter()
        book_rev = defaultdict(list)
        print('counting file:', file_name)
        n_review = 0
        print('current line: ', end='')
        with gzip.open(file_name) as fin:
            for l in fin:
                d = json.loads(l)
                if n_review % 100000==0:
                    print(n_review, end=',')

                #if (n_review+1) % 2000000 == 0:
                #    break
                n_review += 1
                review_count[d['book_id']]+=1
                if d['review_text'].upper().find(phrase)>0:
                    if verbose:
                        print("found phrase",d['book_id'],book_count[d['book_id']])
                    if do_reviews:
                        book_rev[d['book_id']].append(d)
                    book_count[d['book_id']]+=1
        
        # cache reviews for these books
        self.cached_reviews = book_rev
        self.cached_review_count = review_count

        book_count = [k for k in zip(book_count.keys(),book_count.values())]
        best_books = sorted(book_count, key=lambda x: x[1], reverse=True)

        best_book_dict = {x[0]: x[1] for x in best_books}
        self.target_books = best_book_dict

        return self.process_keyphrase_data(review_threshold=review_threshold,hit_threshold=hit_threshold)

    def process_keyphrase_data(self,review_threshold=40,hit_threshold=3):
        normalized_review_count = {}
        unnormalized_review_count = {}
        best_books = self.target_books
        book_rev = self.cached_reviews
        review_count =self.cached_review_count


        for book,num in best_books.items():
            title = greads.titles[book]
            rc= self.cached_review_count[book]
            unnormalized_review_count[book] = num
            #normalized_review_count[book] = num
            #continue
            if rc>review_threshold and num>=hit_threshold:
                normalized_review_count[book] = float(num)/rc * 100.0
            else:
                normalized_review_count[book] = 0
            #print(title, float(num)/rc)
    
        best_normalized = list(zip(normalized_review_count.keys(), normalized_review_count.values()))
        best_normalized = sorted(best_normalized, key=lambda x: x[1], reverse=True)

        return best_books,book_rev,review_count,best_normalized,unnormalized_review_count

#set up globals -- a bit hacky
review_count= pickle.load(open(data_dir + '/review_count.pkl', 'rb'))
greads = goodreads_analysis()
book_title_dict = greads.titles

class book:
    """
    This class is used to represent a book; to make it easier to deal with all the weird ids.
    """
    def __init__(self,gread_id):
        """ init with gread_id of the book, then populate all the other fields """
        self.gread_id = gread_id
        
        global greads, review_count

        self.review_count = review_count[gread_id]

        self.title = greads.titles[gread_id]
        self.csv_id = greads.convert_book_to_csv_id(int(gread_id))
        self.work_id = greads.convert_book_to_work(gread_id)
        self.tokenwork_id = greads.convert_work_to_tokenwork(int(self.work_id))
    def __repr__(self):
        return str( (self.title,self.review_count,self.gread_id,self.csv_id,self.work_id,self.tokenwork_id))


def get_all_books(file_name,fields=None):
    """
    Read in all the books from a file, and return a list of dictionaries that contain the fields specified in *fields*.
    """
    print('counting file:', file_name)
    n_review = 0
    print('current line: ', end='')
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            if fields==None:
                data.append(d)
            else:
                d_new = {}
                for k in fields:
                    d_new[k] = d[k]
                data.append(d_new)
            n_review += 1

            if n_review % 100000 == 0:
                print(n_review, end=',')
            #if n_review % 1 ==0:
            #    print(d)
            #    break
    return data

def make_list_into_dict(_list,key='book_id'):
    new_dict = {}
    keys = list(_list[0].keys())
    for l in _list:
        idx = l[key]
        l.pop(key)
        new_dict[idx] = l
    return new_dict

def process_genres(file_name,num=1e6):
    genre_counter = Counter()
    print('counting file:', file_name)
    n_review = 0
    print('current line: ', end='')
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            e = {}
            genres = d['genres']
            # get the argmax of the genres
            if len(genres) == 0:
                continue
            genremax = max(genres, key=genres.get)

            #for genre in genres:
            #    genre_counter[genre] += 1
            genre_counter[genremax] += 1
            e['book_id'] = d['book_id']
            e['genre'] = genremax
            data.append(e)
            n_review += 1

            if n_review % num == 0:
                print(n_review, end=',')
                
    return data,genre_counter
 
def get_sample(file_name,num=50):
    print('counting file:', file_name)
    n_review = 0
    print('current line: ', end='')
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            data.append(d)
            n_review += 1

            if n_review % num == 0:
                print(n_review, end=',')
                break
    return data



def get_canonical(book_title):
    res = [(k,int(review_count[k[0]])) for k in book_title_dict.items() if k[1].count(book_title)>0]
    res = max(res, key=lambda x: x[1])
    res_dict = {}
    res_dict['book_id'] = res[0][0]
    res_dict['title'] = res[0][1]
    res_dict['review_count'] = res[1]
    print(res_dict)
    return res_dict


def search_for_phrase(phrase):
    best_book_dict,book_rev,review_count,best_normalized,unnormalized_review_count = greads.keyphrase_reviews_in_book(phrase,verbose=False)
    values = sum(list(unnormalized_review_count.values()))
    print(values)

def process_phrase_data(num=50):
    best_book_dict,book_rev,review_count,best_normalized,unnormalized_review_count = greads.process_keyphrase_data(hit_threshold=4)


    num = min(num,len(best_normalized))
    for k in range(num):
        book = best_normalized[k][0]
        num = best_normalized[k][1]
        #book = best_books[k][0]
        title = greads.titles[book]
        #print(title, float(num), best_book_dict[book],book)
        #print(title,num)
        # print title and num, but num as a percentage with 2 decimal places
        print(title, "{:.4f}".format(num),book,unnormalized_review_count[book],review_count[book])
    return best_normalized,review_count

#search_for_phrase("helped me quit")
#_ = process_phrase_data()
def convert_csv_to_work(csv_id):
    book = greads.convert_book_to_gread_id(csv_id)
    work = greads.convert_book_to_work(str(book))
    if work!='':
        work=int(work)
    else:
        work=-1
    return work