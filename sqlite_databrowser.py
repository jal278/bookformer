# %%
import json
import pickle
from collections import defaultdict
from datetime import datetime


def print_in_chunks(review,chunk_size=80):
    print("-"*80)
    chunks = [review[i:i+chunk_size] for i in range(0, len(review), chunk_size)]
    for chunk in chunks:
        print(chunk)
    print("-"*80)


#connect to db
import sqlite3
conn = sqlite3.connect('gread_reviews.db')
cur = conn.cursor()
cur.execute("ATTACH DATABASE 'ratings.db' AS ratings")


# %%
#b_ids = [469571] #Atph
b_ids = [687278] # when things fall apart
#b_ids = [1274]
#b_ids = [46676] # when bad things happen to good people
#b_ids = [7815] # the year of magical thinking
#b_ids = [4069] # man's search for meaning
#b_ids = [31795] # the story of philosophy   

for b_id in b_ids:
    cur = conn.cursor()
    cur.execute('SELECT * FROM reviews WHERE book_id = ?', (b_id,))
    rows = cur.fetchall()
    if len(rows)>1:
        print(b_id,len(rows))
    
r_beg = 60
r_end = 80
print(f"Printing reviews {r_beg}-{r_end} for book_id {b_id} (total reviews: {len(rows)})")
# review 17 of WTFA is deep
for idx in range(r_beg,r_end):
    #print(len(rows))
    print(rows[idx][:3]+rows[idx][4:])

    review = rows[idx][-2]
    print_in_chunks(review)


# user_id is in greads format, need to map to csv format, but need to attach different db (ratings)
# becuase current db is (gread_reviews)
def get_reviews_for_greads_user(user_id):
    cur.execute('''
        SELECT r.*, b.title, b.author, b.book_id
        FROM reviews r
        JOIN books b ON r.book_id = b.book_id
        WHERE r.user_id = ?
        ORDER BY r.date
    ''', (user_id,))
    rows = cur.fetchall()
    return rows

def get_ratings_for_greads_user(user_id):
    # first just get the csv user id 
    cur.execute('SELECT csv_user_id FROM ratings.user_id_map WHERE greads_user_id = ?', (user_id,))
    rows = cur.fetchall()
    csv_user_id = rows[0][0]
    #print(csv_user_id)

    cur.execute('''
        SELECT r.csv_user_id, r.csv_book_id, r.rating, r.time, b.book_id, b.title, b.author
        FROM ratings.ratings r
        JOIN ratings.book_id_map m ON r.csv_book_id = m.csv_book_id
        JOIN books b ON m.greads_book_id = b.book_id
        WHERE r.csv_user_id = ?
        ORDER BY r.time
    ''', (csv_user_id,))
    rows = cur.fetchall()
    return rows

user_id = '1cff79dace43248135560b41e4466904'
revs = get_reviews_for_greads_user(user_id)
ratings = get_ratings_for_greads_user(user_id)

rev_idx = 0
rat_idx = 0

max_reviews_to_print = 5
print("Printing reviews for user",user_id)
print("-"*80)
print("-"*80)

while rev_idx < len(revs) and rat_idx < len(ratings) and rev_idx < max_reviews_to_print:
    rev_time = revs[rev_idx][4]
    rat_time = ratings[rat_idx][3]
    rat_time = datetime.fromtimestamp(rat_time).isoformat()
    if rev_time < rat_time:
        print(revs[rev_idx][4:])
        review = revs[rev_idx][3]
        print_in_chunks(review)
        rev_idx += 1
    else:
        rating = ratings[rat_idx][2]
        title = ratings[rat_idx][5]
        author = ratings[rat_idx][6]
        date = ratings[rat_idx][3]
        date = datetime.fromtimestamp(date).isoformat()
        #print(ratings[rat_idx][:])
        print(f"date: {date} rating: {rating}, title: {title}, author: {author}")
        rat_idx += 1