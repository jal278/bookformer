import pickle
import sample
import numpy as np
from config import data_dir, model_dir

# load from pickle
print("Loading streamlit dict...")
streamlit_dict = pickle.load(open(f'{data_dir}/streamlit_dict.pkl', 'rb'))
booktitles = streamlit_dict['titles']
titles_to_tokens = streamlit_dict['titles_to_token']
tokens_to_titles = {v:k for k,v in titles_to_tokens.items()}

def itos(x):
    """
    Convert a token ID to its string representation.
    
    Args:
        x (int): Token ID to convert
        
    Returns:
        str: String representation of the token:
            - For special tokens (< OFFSET): Returns "[token_name]"
            - For book tokens (>= OFFSET): Returns the book title
            - For unknown tokens: Returns "[unknown]"
    """
    if x<sample.OFFSET:
        title = "[]"
        if x in sample.itos_dict:
            title = "["+sample.itos_dict[x]+"]" 
    else:
        work = x-sample.OFFSET
        if work in tokens_to_titles:
            title = tokens_to_titles[work]
        else:
            title = "[unknown]"
    return title 

print("Loading model...")
model = sample.load_model()

# find a book that includes the phrase
def find_book(phrase):
    """
    Find the first book title containing the given phrase.
    
    Args:
        phrase (str): Phrase to search for in book titles
        
    Returns:
        str: First matching book title, or None if no match found
    """
    for title in booktitles:
        if phrase in title:
            return title
    return None

book = find_book("Siddhartha")
siddhartha_token = titles_to_tokens[book]

book = find_book("Zen and the Art of Motorcycle Maintenance")
motorcycle_token = titles_to_tokens[book]

print(book,motorcycle_token)

#parameters
temperature = 0.8
length = 20


# set up a default context, starting with EOT, then max rating of a specific book (here zen & the art of motorcycle maintenance)
context = [0,5,sample.OFFSET+motorcycle_token]

# sample 10 trajectories that continue the context
for i in range(10):
    print("Generating trajectory ",i)
    y = sample.generate_trajecory(model,context,temperature = temperature,length=length)
    tlist = y.tolist()

    for idx,token in enumerate(tlist):
        title = itos(token)
        #print(token,title)
        print(title, end=';')
    print("\n------")