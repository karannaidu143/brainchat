
from flask import Flask
import datetime
from sqlite3 import Timestamp
import threading
from firebase_admin import credentials, firestore, initialize_app
import firebase_admin



import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer,BartForSequenceClassification, BartTokenizer
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

sentences_b = []
image_urls = []
def image_desc_input(url,desc):
  global sentences_b
  global image_urls
  sentences_b.append(desc)
  image_urls.append(url)
# Sentences we want sentence embeddings for
def image_desc_b(text):
    global sentences_b
    global image_urls
    print(sentences_b,image_urls)
    # query = "me making food for my son"
    query = text
    sentences_b.insert(0,query)

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    # Tokenize sentences
    encoded_input = tokenizer(sentences_b, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        sentences_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(sentences_output, encoded_input['attention_mask'])

    # convert from PyTorch tensor to numpy array
    sentences_mean_pooled = sentence_embeddings.detach().numpy()

    # compute cosine similarity
    similarities = cosine_similarity(
        [sentences_mean_pooled[0]],
        sentences_mean_pooled[1:]
    )

    list1,list2,list3 = zip(*sorted(zip(similarities[0], sentences_b[1:], image_urls),reverse=True))
    output_per_b=list1[0]
    output_text_b=list2[0]
    output_url_b=list3[0]
    print(output_per_b)
    if (output_per_b >(0.90)):
        print('image found sending to server')
        createInFirebase_image(output_text_b,output_url_b)
        return('yes')
    else:
        return('no')




# if query is image query  or not
tokenizer_a = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
model_a = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')


# Initialize Firestore DB
cred = credentials.Certificate('brainchatKey.json')
default_app = initialize_app(cred)
db = firestore.client()
doc_ref = db.collection(u'messages')
doc_ref_persona = db.collection(u'persona')
doc_ref_image= db.collection(u'image')



# work around code 
doc_ref_reset= db.collection(u'reset')

# li=[]
# data stream

# if it is a image query or not (intent classification)
def ml_a(question_a): 
#  question_a = 'display about my cousin'

 candidate_labels_a = ['pictures']  
    
    # run through model pre-trained on MNLI
 for labels in candidate_labels_a:
  input_ids = tokenizer_a.encode(question_a, labels, return_tensors='pt')
  logits = model_a(input_ids)[0]

  # we throw away "neutral" (dim 1) and take the probability of
  # "entailment" (2) as the probability of the label being true 
  entail_contradiction_logits = logits[:,[0,2]]
  probs = entail_contradiction_logits.softmax(dim=1)
  true_prob = probs[:,1].item() * 100
  print(true_prob)
  print(f'Probability that the label is true: {true_prob:0.2f}%')
  if(int(true_prob)>80):
      return('yes')
  else:
      return('no')

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output
parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)")  # anything besides gpt2 will load openai-gpt
parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--temperature", type=float, default=0.4,help="Sampling softmax temperature")
parser.add_argument("--top_k", type=int, default=12, help="Filter top-k tokens before sampling (<=0: no filtering)")
parser.add_argument("--top_p", type=float, default=0.99, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.info(pformat(args))

if args.model_checkpoint == "":
    if args.model == 'gpt2':
        raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
    else:
        args.model_checkpoint = download_pretrained_model()


if args.seed != 0:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


logger.info("Get pretrained model and tokenizer")
tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
model = model_class.from_pretrained(args.model_checkpoint)
model.to(args.device)
add_special_tokens_(model, tokenizer)

history = []
per=[]
# userInfo=['my name is karan','i am 40 years old','i went scuba diving last week it was super fun','i received an award yesterday for best paper presentation','my son`s birthday is on 25th july']
# per=[]
# for i in userInfo:
#             per.append(tokenizer.encode(i))
def persona_update(text):
    global per
    per.append(tokenizer.encode(text+'.'))


def reset():
    global history
    global per    
    history=[]
    per=[]
    print("reset the server")
def reset_chat():
    global per    
    per=[]
    print("reseted the chat")
    
def ml(text):
    global history
    global per
    # print(tokenizer.decode(chain(*per)))

    raw_text = text
    #raw_text = input(">>> ")
    # if not raw_text:
    #     print('Text should not be empty!')
    #     raw_text = input(">>> ")

    # print(tokenizer.decode(chain(*per)))
    history.append(tokenizer.encode(raw_text))
    # print(tokenizer.decode(chain(*history)))
    with torch.no_grad():
      out_ids = sample_sequence(per, history, tokenizer, model, args)
    history.append(out_ids)
    
    history = history[-(2*args.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    print(out_text)
    createInFirebase(out_text)
    # output text should be updated to the user

callback_done = threading.Event()

def on_snapshot(doc_snapshot, changes, read_time):
  li=[]
  if doc_snapshot:
    for doc in doc_snapshot:
        #  print(f'Received document snapshot: {doc.to_dict()}')    
         li.append(doc.to_dict())
    # print(li)
    li.sort(key=lambda x:x['timestamp'])
    # print(li)
    # print(li.pop()['sender'])
    latestText=li.pop()
    # print(latestText)
    sender=latestText['sender']
    text=latestText['text']
    print(sender)
    if sender!='bot@gmail.com':
        # print(text)
      
      is_query_image=ml_a(text)
      if(is_query_image=='yes'):
          output=image_desc_b(text)
          if (output=='no'):
            #   if the accuracy of the image  is low, then pass to ml model
              ml(text)
          else:
              print('workflow done')    

      else:
          ml(text)
                
    callback_done.set()


    #Code for watching personality add on
callback_persona_done = threading.Event()
def on_snapshot_persona(doc_snapshot_persona, changes, read_time):
  print("second function ")
  li=[]
  if doc_snapshot_persona:
    for doc in doc_snapshot_persona:

        #  print(f'Received document snapshot: {doc.to_dict()}')    
         li.append(doc.to_dict())
    # print(li)
    li.sort(key=lambda x:x['timestamp'])
    # print(li)
    # print(li.pop()['sender'])
    latestText=li.pop()
    # print(latestText)
    
    text=latestText['text']
    
    # print(text)
    persona_update(text)    

    callback_persona_done.set() 

# to reset the datapip install sentence-transformerspip install sentence-transformers
callback_reset_done = threading.Event()
def on_snapshot_reset(doc_snapshot_reset, changes, read_time):
  print("3rd function ")
  li=[]
  if doc_snapshot_reset:
    for doc in doc_snapshot_reset:

        #  print(f'Received document snapshot: {doc.to_dict()}')    
         li.append(doc.to_dict())
    # print(li)
    li.sort(key=lambda x:x['timestamp'])
    # print(li)
    # print(li.pop()['sender'])
    latestText=li.pop()
    # print(latestText)
    
    text=latestText['text']  
    print(text)
    if(text=='clear the chat'):
         reset_chat()     
    else:    
       reset()

    callback_reset_done.set() 

def on_snapshot_image(doc_snapshot_image, changes, read_time):
  print("3rd function ")
  li=[]
  if doc_snapshot_image:
    for doc in doc_snapshot_image:

        #  print(f'Received document snapshot: {doc.to_dict()}')    
         li.append(doc.to_dict())
    # print(li)
    li.sort(key=lambda x:x['timestamp'])
    # print(li)
    # print(li.pop()['sender'])
    latestText=li.pop()
    # print(latestText)
    imageUrl=latestText['imageUrl']
    text=latestText['text']  
    print(text)
    image_desc_input(imageUrl,text)    

    callback_reset_done.set() 


# Watch the document
doc_watch = doc_ref.on_snapshot(on_snapshot)

#Watch changes in personality
doc_watch = doc_ref_persona.on_snapshot(on_snapshot_persona)

# the value is reset every time a new doc is created
doc_watch = doc_ref_reset.on_snapshot(on_snapshot_reset)

# watch if imageurll and text is updated
doc_watch = doc_ref_image.on_snapshot(on_snapshot_image)
def createInFirebase(response):
    try:
        doc_ref.document().set({
          "sender": "bot@gmail.com",
          "text": response,
          "image":'',
          "timestamp": firebase_admin.firestore.SERVER_TIMESTAMP

      })
        print('created')

    except Exception as e:
        print(f'An Error Occured: {e}')
def createInFirebase_image(response,url):
    try:
        doc_ref.document().set({
          "sender": "bot@gmail.com",
          "text": response,
          "image":url,
          "timestamp": firebase_admin.firestore.SERVER_TIMESTAMP
      })
        print('created')

    except Exception as e:
        print(f'An Error Occured: {e}')
      


    # logger.info("Sample a personality")
    # dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    
    # personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
    # personality = random.choice(personalities)
    # # print(personality)
    # logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

app = Flask(__name__)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=8000,use_reloader=False)

