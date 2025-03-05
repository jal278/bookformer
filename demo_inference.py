import pickle
import sample
import numpy as np


# load from pickle
print("Loading streamlit dict...")
streamlit_dict = pickle.load(open('streamlit_dict.pkl', 'rb'))
booktitles = streamlit_dict['titles']
titles_to_tokens = streamlit_dict['titles_to_token']
tokens_to_titles = {v:k for k,v in titles_to_tokens.items()}

addtl_tokens_dict= {}
addtl_tokens_dict[0] ='[EOT]'
addtl_tokens_dict[1] = 'R1'
addtl_tokens_dict[2] = 'R2'
addtl_tokens_dict[3] = 'R3'
addtl_tokens_dict[4] = 'R4'
addtl_tokens_dict[5] = 'R5'





#parameters
temperature = 0.8
length = 20

def itos(x):
    if x<sample.OFFSET:
        title = "[]"
        if x in addtl_tokens_dict:
            title = addtl_tokens_dict[x]
    else:
        work = x-sample.OFFSET
        if work in tokens_to_titles:
            title = tokens_to_titles[work]
        else:
            title = "[unknown]"
    return title 

print("Loading model...")
model = sample.load_model()

context = [0]

print("Generating trajectory...")
y = sample.generate_trajecory(model,context,temperature = temperature,length=length)
tlist = y.tolist()

for idx,token in enumerate(tlist):
    title = itos(token)
    print(token,title)