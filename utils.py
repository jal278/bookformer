
# code for read in ratings into memory...
if False:
    fname = base_dir + '/interactions_proc_filtered_new_merged.csv'

    ratings = pd.read_csv(
            fname,
            sep=",",
            usecols=[0, 1, 2,3],
            header=0,
            dtype={'user':np.uint32,'item':np.uint16,'rating':np.uint8,'time':np.uint32}
        )
    ratings.sort_values(by=['user','time'], inplace=True)
# %%
if True:
    ratings['work'] = ratings['item'].map(convert_csv_to_work).astype(np.uint32)

# %% read book_dict into memory...
import pickle
book_dict = pickle.load(open('book_detail_dict.pkl', 'rb'))
author_dict = pickle.load(open('author_detail_dict.pkl', 'rb'))
# %%


# %%
#sample = get_sample(base_dir + '/goodreads_books.json.gz')

# %%
#sample = get_sample(base_dir + '/goodreads_books.json.gz')
sample = get_sample(base_dir + '/goodreads_book_works.json.gz')

if False:
    genres,genrecounter = process_genres(base_dir + '/goodreads_book_genres_initial.json.gz')
    book_to_genre_map = {}
    for item in genres:
        book_to_genre_map[item['book_id']] = item['genre']
    pickle.dump(book_to_genre_map,open(base_dir+'/book_to_genre_map.pkl','wb'))

if True:
    res = get_all_books(base_dir + '/goodreads_book_works.json.gz',fields=['work_id','best_book_id','original_title','rating_dist','ratings_count'])
#book_to_genre_map = pickle.load(open(base_dir+'/book_to_genre_map.pkl','rb'))

if False:
    #res = get_all_books(base_dir + '/goodreads_book_authors.json.gz',fields=['author_id','average_rating','name','ratings_count'])
    #author_dict = make_list_into_dict(res,key='author_id')
    #res.sort(key=lambda x: float(x['average_rating'] if int(x['ratings_count'])>5000 else 0),reverse=True)

    res = get_all_books(base_dir + '/goodreads_books.json.gz',fields=['book_id','title','publication_year','authors','isbn','asin','kindle_asin','average_rating','ratings_count'])
    book_dict = make_list_into_dict(res)
    #pickle.dump(book_dict, open('book_detail_dict.pkl', 'wb'))
    #pickle.dump(author_dict, open('author_detail_dict.pkl', 'wb'))
# %%
work_dict = make_list_into_dict(res,key='work_id')


# %%
import requests
import json

# set up the request parameters
def look_up_asin(isbn):
    params = {
    'api_key': '218B20E029284839B21B4483B1210BFA',
    'type': 'search',
    'amazon_domain': 'amazon.com',
    'search_term': '0312853122',
    'output': 'json',
    'associate_id': 'flourishin0c7-20'
    }

    # make the http GET request to ASIN Data API
    api_result = requests.get('https://api.asindataapi.com/request', params)

    # print the JSON response from ASIN Data API
    print(json.dumps(api_result.json()))
