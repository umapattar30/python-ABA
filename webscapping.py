pip install nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
import nltk
df = pd.read_csv('Reviews.csv')
df.head()
df = df.head(5000)
df['Text'].values[0]
df.shape
ax = df['Score'].value_counts().sort_index().plot(kind = 'bar',
                                             title = 'Counts of Reviews by Stars',
                                             figsize = (10,5))
ax.set_xlabel('Review Stars')
plt.show()
example = df['Text'][50]
print(example)
import nltk
nltk.download('punkt')
tokens = nltk.word_tokenize(example)
tokens[:10]
import nltk
nltk.download('averaged_perceptron_tagger')
tagged = nltk.pos_tag(tokens)
tagged[:10]
import nltk
nltk.download('maxent_ne_chunker')
import nltk
nltk.download('words')
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
sia
sia.polarity_scores('I am so nervous')
sia.polarity_scores('I am so happy')
results = {}
for i, row in tqdm(df.iterrows(), total = len(df)):
  text = row['Text']
  myid = row['Id']
  results[myid] = sia.polarity_scores(text)
pd.DataFrame(results)
vaders = pd.DataFrame(results).T
vaders.head()
vaders = vaders.reset_index().rename(columns = {'index':'Id'})
vaders = vaders.merge(df,how = 'left')
vaders.head()
ax = sns.barplot(data = vaders, x = 'Score', y = 'compound')

ax.set_title('Compound score by Amazon Star Reviews')
plt.show()
sns.barplot(data=vaders, x = 'Score', y = 'pos')
fig , axs = plt.subplots(1,3,figsize = (15,5))
sns.barplot(data = vaders, x='Score', y= 'pos', ax=axs[0])
sns.barplot(data = vaders, x='Score', y= 'neu', ax=axs[1])
sns.barplot(data = vaders, x='Score', y= 'neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()