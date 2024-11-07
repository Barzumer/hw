import numpy as np
import pandas as pd
import re
import nltk
import string
df = pd.read_csv("hf://datasets/andidu/paraphrase-ru-reviews/dataset.csv", encoding='UTF8', sep="\t")
df.head(10)
df.iloc[[1]]
df.tail(3)
df['text_lower']  = df['id,O,P'].str.lower()
df['text_lower'].head()
df.head()
df['text_punct'] = df['id,O,P'].str.replace('[^\w\s]','')
df['text_punct'].head()
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('russian'))
def stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
df["text_stop"] = df["text_punct"].apply(stopwords)
df["text_stop"].head()
from collections import Counter
cnt = Counter()
for text in df["text_stop"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)
freq = set([w for (w, wc) in cnt.most_common(10)])
# функция удаления слов
def freqwords(text):
    return " ".join([word for word in str(text).split() if word not 
in freq])
# применение функции
df["text_common"] = df["text_stop"].apply(freqwords)
df["text_common"].head()
#Удаление низкочастотных слов
freq = pd.Series(' '.join(df['text_common']).split()).value_counts()[-10:] # 10 rare words
freq = list(freq.index)
df['text_rare'] = df['text_common'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df['text_rare'].head()
# Удаление эмодзи
def emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
#passing the emoji function to 'text_rare'
df['text_rare'] = df['text_rare'].apply(emoji)
#Удаление цифр
df['text_nonum'] = df['text_common'].str.replace('\d+', '') 
df['text_nonum'].head()
#Удаление URL
# Function for url's
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
# Examples
text = "This is my website, https://www.link.com"
remove_urls(text)
#Passing the function to 'text_rare'
df['text_rare'] = df['text_rare'].apply(remove_urls)
#Удаление HTML-тегов
from bs4 import BeautifulSoup
#Function for removing html
def html(text):
    return BeautifulSoup(text, "lxml").text
# Examples
text = """<div>
<h1> Это </h1>
<p> странца</p>
<a href="https://www.link.com/"> пример</a>
</div>
"""
print(html(text))
# Passing the function to 'text_rare'
df['text_rare'] = df['text_rare'].apply(html)
#Токенизация
def tokenization(text):
    text = re.split('\W+', text)
    return text
# Passing the function to 'text_rare' and store into'text_token'
df['text_token'] = df['text_rare'].apply(lambda x: tokenization(x.lower()))
df[['text_token']].head()
#Можно выполнить при помощи NLTK. Заодно проведем лемматизацию
from pymorphy3 import MorphAnalyzer
from nltk import sent_tokenize, word_tokenize, regexp_tokenize

def tokenize_lemmas(sent, pat=r"(?u)\b\w\w+\b", morph=MorphAnalyzer()):
    return [morph.parse(tok)[0].normal_form 
            for tok in regexp_tokenize(sent, pat)]
df["text_lemm"] = df["text_rare"].map(lambda x: " ".join(tokenize_lemmas(x)))
df[['text_lemm']].head()
#Исходя из цели автоматической обработки текста, выберем необходимые варианты предобработки текста. 
#Например, для классификации можно воспользоваться нижним регистром, удалением  пунктуации, ссылок, стоп-слов, 
#произвести лемматизацию и токенизацию
##ИЛИ воспользоваться только удалением ссылок. Т.к. это актуальнее для НС, а выбор предобработки сделал для Машинного обученя
df['text_ready']  = df['id,O,P'].str.lower()
df['text_ready'] = df['text_ready'].str.replace('\d+', '') 
df['text_ready'] = df['text_ready'].str.replace('[^\w\s]','')
df["text_ready"] = df["text_ready"].apply(stopwords)
df['text_ready'] = df['text_ready'].apply(emoji)
df['text_ready'] = df['text_ready'].apply(remove_urls)
df['text_ready'] = df['text_ready'].apply(html)
df["text_ready"] = df["text_ready"].map(lambda x: " ".join(tokenize_lemmas(x)))
df[['text_ready']].head()
import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment', return_dict=True)


def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    return predicted

#txt = df.iloc[2,2]
txt = "Хорошая погода! Наконец-то тепло!"
print(txt)

predict(txt)
