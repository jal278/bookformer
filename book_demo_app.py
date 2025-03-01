# seeing like a state
# alchemy
# already free

import streamlit as st
import pickle
import transformational_sample
import numpy as np


# load from pickle
streamlit_dict = pickle.load(open('streamlit_dict.pkl', 'rb'))
booktitles = streamlit_dict['titles']
titles_to_tokens = streamlit_dict['titles_to_token']
tokens_to_titles = {v:k for k,v in titles_to_tokens.items()}

temperature = 0.8

st.write("""
         # Book Recs!
         """)


options = st.multiselect(
    'Choose Books',
    booktitles,
    [])

cml = st.checkbox('Change My Life')
permap = st.checkbox('PERMA P')
permae = st.checkbox('PERMA E')
surprise = st.checkbox('Surprise')
erotic = st.checkbox('Erotic')
best = st.checkbox('Best')
worst = st.checkbox('Worst')
gift = st.checkbox('Gift')
weird = st.checkbox('Weird')

rating_5 = st.checkbox('Highly Rated')
rating_1 = st.checkbox('Low Rated')

accentuate_end = st.checkbox('Accentuate End')
accentuate_history = st.checkbox('Accentuate History')

accent_number = st.slider('Accentuate Number', 0, 4, 1)
temperature = st.slider('Temperature', 0.0, 1.0, 0.8)

st.write('You selected:', options)

recs_clicked = st.button('Recs')
traj_clicked = st.button('Simulate Trajectory')

# text box
traj_text = st.text_area("Trajectory", "")

def itos(x):
    if x<transformational_sample.OFFSET:
        title = "[]"
    else:
        work = x-transformational_sample.OFFSET
        if work in tokens_to_titles:
            title = tokens_to_titles[work]
        else:
            title = "[unknown]"
    return title 


if recs_clicked or traj_clicked:
    if "model" not in st.session_state.keys():
        st.session_state["model"] = transformational_sample.load_model()
    model = st.session_state["model"]
    all_context = [0]

    for item in options:
        context = []
        if transformational_sample._do_ratings:
            context.append(5)
        context.append(titles_to_tokens[item]+transformational_sample.OFFSET)
        all_context.append(context)

    context = []
    if cml:
        for i in range(accent_number):
            context.append(transformational_sample.ACCENT)
        context.append(transformational_sample.CML)
    if permap:
        for i in range(accent_number):
            context.append(transformational_sample.ACCENT)
        context.append(transformational_sample.PERMAP)
    if permae:
        for i in range(accent_number):
            context.append(transformational_sample.ACCENT)
        context.append(transformational_sample.PERMAE)
    if surprise:
        for i in range(accent_number):
            context.append(transformational_sample.ACCENT)
        context.append(transformational_sample.SURPRISE)

    if erotic:
        for i in range(accent_number):
            context.append(transformational_sample.ACCENT)
        context.append(transformational_sample.EROTIC)

    if best:
        for i in range(accent_number):
            context.append(transformational_sample.ACCENT)
        context.append(transformational_sample.BEST)

    if worst:
        for i in range(accent_number):
            context.append(transformational_sample.ACCENT)
        context.append(transformational_sample.WORST)

    if gift:
        for i in range(accent_number):
            context.append(transformational_sample.ACCENT)
        context.append(transformational_sample.GIFT)

    if weird:
        for i in range(accent_number):
            context.append(transformational_sample.ACCENT)
        context.append(transformational_sample.WEIRD)

    if rating_5:
        context.append(5)
    if rating_1:
        context.append(1)

    all_context.append(context)

    if traj_text:
        all_context = []
        traj_list = traj_text.split(",")
        for item in traj_list:
            all_context.append(int(item))
    st.write(all_context)

    def unfold_context(context):
        new_context = []
        for item in context:
            # if item is a list, then add each item
            if isinstance(item,list):
                for subitem in item:
                    new_context.append(subitem)
            else:
                new_context.append(item)
        return new_context
    # create recs
    if recs_clicked:
        context = all_context
        if accentuate_end or accentuate_history:
            if not transformational_sample._do_ratings:
                if accentuate_end:
                    dummy_context = context[:-1]
                if accentuate_history:
                    dummy_context = [0] + context[-1:]
            else:
                if accentuate_end:
                    dummy_context = context[:-1] + [context[-1][-1:]]
                if accentuate_history:
                    dummy_context = context[0:1] + context[-1:]
            st.write(dummy_context)


        
            context = unfold_context(context)
            dummy_context = unfold_context(dummy_context)

            probs,probs_un,probs_0 = transformational_sample.accentuated_recs(model,context,dummy_context)
        else:
            context = unfold_context(context)
            probs = transformational_sample.get_next_book_probs(model,context,zero_out_prev=True,temperature = 1.0)

        num = 20
        top = list(np.argsort(probs)[-num:])
        top.reverse()
        token,t_probs = top,probs[top]

        for i in range(len(token)):
            title = itos(token[i])
            st.write(token[i],title,t_probs[i])
    
    # simulate trajectory
    if traj_clicked:
        context = all_context
        context = unfold_context(context)

        y = transformational_sample.generate_trajecory(model,context,temperature = temperature,length=20)
        tlist = y.tolist()
        st.write(tlist)
        for idx,token in enumerate(tlist):
            title = itos(token)
            st.write(token,title)
            if idx==len(context):
                st.write("------")
            
 

