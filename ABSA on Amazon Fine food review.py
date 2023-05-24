import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import wordcloud
import tensorflow as tf

import os

warnings.filterwarnings("ignore")
  
plt.style.use('seaborn')

import nltk
# Read in data
df = pd.read_csv(r'Reviews.csv')

print(df.shape)
df = df.head(1000)
print(df.shape)

df.head()

def add_value_labels(ax, spacing=5):
    
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
            
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        
        space = spacing
        # Vertical alignment for positive values
        
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'
            
        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,
                      # Use `label` as label
                      
            (x_value, y_value), 
                        # Place label at end of the bar
                        
            xytext=(0, space),  
                        # Vertically shift label by `space`
                        
            textcoords="offset points",
                        # Interpret `xytext` as offset in points
                        
            ha='center',      
                       # Horizontally center label
                       
            va=va)                    
                      # Vertically align label differently for
                                       
                      # positive and negative values.

ax = df['Score'].value_counts().sort_index() .plot(kind='bar',figsize=(15, 6))

ax.set_xlabel('Review Stars',fontsize=15,fontweight='bold')

ax.set_title('Count of Reviews by Score',fontsize=25,fontweight='bold')

add_value_labels(ax)

plt.show()
# Select any one Tweet
 
# example = df['Text'].values[0]
example = df['Text'][30]
print(example)

# Tokenization 
tokens = nltk.word_tokenize(example)
tokens[:15]

# Part Of speech Tagging.
tagged = nltk.pos_tag(tokens)
tagged[:15]

# To chunk the given list of tokens , so it takes tokens and actually will group them into chunks of text.
entities = nltk.chunk.ne_chunk(tagged)

entities.pprint()
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

# let's Test our example text polarity score.
sia.polarity_scores(example)
#To print the cluster and common words
print("Common Words Are")

common_words=''
for i in df.Text:
    i = str(i)
    
    tokens = i.split()
    
    common_words += " ".join(tokens)+" "
wordcloud = wordcloud.WordCloud().generate(common_words)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


plt.show() 

# Run the polarity score on the entire dataset.
result = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    result[myid] = sia.polarity_scores(text)
result
vaders = pd.DataFrame(result).T

vaders = vaders.reset_index().rename(columns={'index': 'Id'})

vaders = vaders.merge(df, how='left')

# Now we have sentiment score and metadata.

# Let's check  sentimental score and Metadata head values once.
vaders.head()
def vader_sntms(value):
    if value >= 0.05:
        return "Positive"
    elif value <= -0.05:
        return "Negative"
    else:
        return "Neutral"
x = vaders['compound'].apply(vader_sntms)

vaders.insert(5,'vader_sntms',x)

vaders.head()
plt.figure(figsize=(15,6))

ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review',fontsize=25,fontweight='bold')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])

sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])

sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])

axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')

plt.tight_layout()

plt.show()

vaders['vader_sntms'].value_counts().plot.pie(explode=(0.05,0.05,0.05),colors=['#369cef','#36efd6','#b8ef36'],autopct='%1.1f%%',shadow=True,figsize=(15,6),fontsize=15,labels=None)

plt.title("Percentage Distribution of Sentiments",fontsize=25,fontweight='bold')
labels=vaders['vader_sntms'].value_counts().index.tolist()

plt.legend(bbox_to_anchor=(0.85, 1), loc='upper left', labels=labels)

plt.show()

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# VADER results on example

print(example)

vader_example = sia.polarity_scores(example)

print("\n",vader_example)

# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')

output = model(**encoded_text)

scores = output[0][0].detach().numpy()

scores = softmax(scores)

scores_dict = {
    'roberta_neg' : scores[0],
    
    'roberta_neu' : scores[1],
    
    'roberta_pos' : scores[2]
}

print(scores_dict)

both = {**vader_example , **scores_dict}

both
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        
        'roberta_neu' : scores[1],
        
        'roberta_pos' : scores[2]
    }
    return scores_dict

res = {}

for i, row in tqdm(df.iterrows(), total=len(df)):
    
    try:
        text = row['Text']
        
        myid = row['Id']
        
        vader_result = sia.polarity_scores(text)
        
        vader_result_rename = {}
        
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
        
    except RuntimeError:
        print(f'Broke for id {myid}')
        results_df = pd.DataFrame(res).T
        
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')
results_df.head()


sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')

plt.show()

# Positive sentiment 1-Star view.
results_df.query('Score == 1').sort_values('roberta_pos', ascending=False)['Text'].values[0]

# Positive sentiment 1-Star view.
results_df.query('Score == 1').sort_values('vader_pos', ascending=False)['Text'].values[0]

# nevative sentiment 5-Star view.
results_df.query('Score == 5').sort_values('roberta_neg', ascending=False)['Text'].values[0]

# nevative sentiment 5-Star view.
results_df.query('Score == 5').sort_values('vader_neg', ascending=False)['Text'].values[0]
df1 = df

df1
from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    
    if sentiment_polarity > 0:
        sentiment_label = 'Positive'
        
    elif sentiment_polarity < 0:
        sentiment_label = 'Negative'
        
    else:
        sentiment_label = 'Neutral'
        
    result = {'polarity':sentiment_polarity,
              'subjectivity':sentiment_subjectivity,
              'sentiment':sentiment_label}
    
    return result

df1['sentiment_results'] = df1['Text'].apply(get_sentiment)

df1['sentiment_results'].head()

df1 = df1.join(pd.json_normalize(df1['sentiment_results']))

df1.head()
plt.figure(figsize=(15,6))

ax = sns.barplot(data=df1, x='Score', y='polarity')

ax.set_title('Polarity Score by Amazon Star Review',fontsize=25,fontweight='bold')

ax.set_xlabel("Amazon Reviews",fontsize=15,fontweight='bold')

ax.set_ylabel("Polarity Score",fontsize=15,fontweight='bold')
plt.show()

df1['sentiment'].value_counts().plot.pie(explode=(0.05,0.05,0.05),colors=['#369cef','#36efd6','#b8ef36'],autopct='%1.1f%%',shadow=True,figsize=(15,6),fontsize=15,labels=None)

plt.title("Percentage Distribution of Sentiments",fontsize=25,fontweight='bold')

labels=df1['sentiment'].value_counts().index.tolist()

plt.legend(bbox_to_anchor=(0.85, 1), loc='upper left', labels=labels)
plt.show()

plt.figure(figsize=(15,6))

ax = sns.barplot(data=df1, x='Score', y='polarity',hue='sentiment')

ax.set_title('Polarity Score by Amazon Star Review',fontsize=25,fontweight='bold')

ax.set_xlabel("Amazon Reviews",fontsize=15,fontweight='bold')

ax.set_ylabel("Polarity Score",fontsize=15,fontweight='bold')
plt.show()

plt.figure(figsize=(15,6))

ax = sns.lineplot(data=df1, x='Score', y='polarity',hue='sentiment')

ax.set_title('Polarity Score by Amazon Star Review',fontsize=25,fontweight='bold')

ax.set_xlabel("Amazon Reviews",fontsize=15,fontweight='bold')

ax.set_ylabel("Polarity Score",fontsize=15,fontweight='bold')
plt.show()

# using HuggingFace's transformers library to perform sentiment analysis on Netflix customer reviews:

import torch
from transformers import pipeline, set_seed

set_seed(42)

sentiment_pipeline = pipeline("sentiment-analysis")

reviews = [
    "I love this movie!", 
    "This movie is not good.", 
    "This movie is average.", 
    "I don't like this movie."
]

results = [sentiment_pipeline(review)[0]["label"] for review in reviews]

print(results)

result = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        
        myid = row['Id']
        
        result[myid] = sentiment_pipeline(text)
        
    except RuntimeError:
        print(f'Broke for id {myid}')
        
        transformer_df1 = pd.DataFrame(result).T
        
transformer_df1.head()

transformer_df1 = transformer_df1.join(pd.json_normalize(transformer_df1[0]))
transformer_df1.head()

transformer_df1.drop(0,axis=1,inplace=True)
transformer_df1.head()

transformer_df1 = transformer_df1.reset_index().rename(columns={'index':'Id'})
transformer_df1.head()

transformer_df1 = transformer_df1.merge(df,how='left')
transformer_df1.head()

plt.figure(figsize=(15,6))

ax = sns.barplot(data=transformer_df1, x='Score', y='score')

ax.set_title('Sentiment Score by Amazon Star Review',fontsize=25,fontweight='bold')

ax.set_xlabel("Amazon Reviews",fontsize=15,fontweight='bold')

ax.set_ylabel("Sentiment Score",fontsize=15,fontweight='bold')
plt.show()

transformer_df1['label'].value_counts().plot.pie(explode=(0.05,0.05),colors=['#369cef','#36efd6'],autopct='%1.1f%%',shadow=True,figsize=(15,6),fontsize=15,labels=None)

plt.title("Percentage Distribution of Sentiments",fontsize=25,fontweight='bold')

labels=transformer_df1['label'].value_counts().index.tolist()

plt.legend(bbox_to_anchor=(0.85, 1), loc='upper left', labels=labels)
plt.show()

plt.figure(figsize=(15,6))

ax = sns.barplot(data=transformer_df1, x='Score', y='score',hue='label')

ax.set_title('Sentiment Score by Amazon Star Review',fontsize=25,fontweight='bold')

ax.set_xlabel("Amazon Reviews",fontsize=15,fontweight='bold')

ax.set_ylabel("Sentiment Score",fontsize=15,fontweight='bold')
plt.show()

plt.figure(figsize=(15,6))


ax = sns.lineplot(data=transformer_df1, x='Score', y='score',hue='label')

ax.set_title('Sentiment Score by Amazon Star Review',fontsize=25,fontweight='bold')

ax.set_xlabel("Amazon Reviews",fontsize=15,fontweight='bold')

ax.set_ylabel("Sentiment Score",fontsize=15,fontweight='bold')
plt.show()