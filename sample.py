# %%
# nanoGPT loader
import numpy as np
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
# reload model module
from importlib import reload
import model
from model import GPTConfig, GPT

_mname = 'norate-large'
_mname = 'v1'
_mname = 'v2'

if _mname == 'vanilla':
    _do_ratings = True
    OFFSET = 12 #6
    EOT = 0
    CML = 6
    DO = 7
    PERMAP = 8
    PERMAE = 9
    SURPRISE = 10
    NORATE = 11
elif _mname == 'norate-large':
    _do_ratings = False
    OFFSET = 7
    EOT = 0
    CML = 1
    DO = 2
    PERMAP = 3
    PERMAE = 4
    SURPRISE = 5
    NORATE = 6
elif _mname == 'v1':
    _do_ratings = True
    OFFSET = 19
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
    write_ratings = True
    write_work = True
elif _mname == 'v2':
    _do_ratings = True
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
    REVERSE = 19
    FIM = 20
    FIM2 = 21
    write_ratings = True
    write_work = True


device = "cpu"
dtype = "float16"
compile = False
seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


model_dir = "./out_colab_v2_100m_reg"



def load_model():
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(model_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


# %%

from torch.nn import functional as F

def decode_token(token):
    if token>=OFFSET:
        token = token-OFFSET
        book = greads.convert_tokenwork_to_book(token)
        title = greads.titles[book]
        return title
    else:
        return str(token)
    
def make_context(context_books,works=True):
    context = []
    print(context)
    for book in context_books:
        if isinstance(book,int):
            item = book
        else:
            _item = int(book['book_id'])
            if not works:
                item = greads.convert_book_to_csv_id(_item)+OFFSET
            else:
                work = greads.convert_book_to_work(str(_item))
                item = greads.work_to_tokenwork_map[int(work)]+OFFSET
        context.append(item)
    return context

def generate_trajecory(model,context,temperature = 0.3,length=5):
    with torch.no_grad():
        # sample from the model...
        x = torch.tensor(context, dtype=torch.long)[None,...]
        y = model.generate(x, length, temperature=temperature)[0]
        #completion = ' '.join([str(int(i)) for i in y])
        #print(completion)
    return y

def get_next_book_probs(model,context,zero_out_prev=False,temperature = 0.3):
    with torch.no_grad():
        # sample from the model...
        x = torch.tensor(context, dtype=torch.long)[None,...]
        logits,_ = model(x)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        #completion = ' '.join([str(int(i)) for i in y])
        #print(completion)

    probs = probs.numpy()[0]
    if zero_out_prev:
        for i in context:
            probs[i] = 0
        # renormalize
        probs /= probs.sum()

    return probs

def get_all_book_probs(model,context,temperature = 0.3):
    with torch.no_grad():
        # sample from the model...
        x = torch.tensor(context, dtype=torch.long)[None,...]
        logits,_ = model(x,)
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        #completion = ' '.join([str(int(i)) for i in y])
        #print(completion)
    return probs

def print_top_tokens(probs,num=30):
    top = list(np.argsort(probs)[-num:])
    top.reverse()
    token,t_probs = top,probs[top]
    for i in range(len(token)):
        title = decode_token(token[i])
        print(token[i],title,t_probs[i])

def accentuated_recs(model,context,dummy=[CML]):
    dummy_context = make_context(dummy)

    probs_0 = get_next_book_probs(model,dummy_context,temperature=1.0,zero_out_prev=True)
    probs = get_next_book_probs(model,context,temperature=1.0,zero_out_prev=True)

    accentuated_probs = probs - probs_0
    # now clip less than zero and renormalize
    accentuated_probs = np.clip(accentuated_probs,0,1)

    accentuated_probs /= accentuated_probs.sum()
    return accentuated_probs,probs,probs_0
# %%
if __name__=='__main__':
    model = load_model()

    cy = get_canonical("Autobiography of a Yogi")


    # %%
    temperature = 1.0
    #context_books = [lobster,3543,love,19730,smoke,mithc,bluets,delta]

    start_ids = make_context([EOT,3,209,3,242,5,ublob,5,lobster,5,tbt,PERMAP,5])
    start_ids = make_context([EOT,5,294])
    # %%
    accentuated,probs,probs_0 = accentuated_recs(start_ids,dummy=[EOT,CML,5])
    print_top_tokens(accentuated)
    print('----------------')
    print_top_tokens(probs)
    print('----------------')
    print_top_tokens(probs_0)
    # %%
    probs = get_next_book_probs(start_ids,temperature=temperature)
    print_top_tokens(probs)
    # %%
    print(start_ids)
    y = generate_trajecory(start_ids,temperature=1.0,length=10)
    tlist = y.tolist()
    for token in tlist:
        print(decode_token(token))
    # %%
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    with torch.no_grad():
        with ctx:
            y = model.generate(x, 10, temperature=temperature)
            tlist = y[0].tolist()
            for token in tlist:
                print(decode_token(token))
            print('---------------')
