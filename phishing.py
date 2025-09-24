import pandas as pd                              #DATA MANIPULATION AND ANALYSIS
import numpy as np                               #NEUMERICAL OPERATIONS
import matplotlib.pyplot as plt                 #DATA VISUALIZATION
from nltk.tokenize import RegexpTokenizer       #TEXT SPLITTING, FOR TOKENIZATION
from nltk.stem.snowball import SnowballStemmer  #normalizes the variations to a single root word
from wordcloud import WordCloud                 #generate word cloud images.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
import pickle
#TOKENIZATION
df=pd.read_csv('dataset/phishing_site_urls.csv')            #READ FROM DATASET
tokenizer=RegexpTokenizer(r'[A-Za-z]+')                      #RegexpTokenizer OBJ CREATION
df['text_tokenized']=df.URL.map(lambda t: tokenizer.tokenize(t))        #converts raw URLs into a format suitable for further analysis

#STEMMING
stemmer=SnowballStemmer('english')
df['text_stemmed']=df['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l] )
df['text']=df['text_tokenized'].map(lambda l: ' '.join(l))

#Slice
good_sites=df[df.Label=='good']
bad_sites=df[df.Label=='bad']

def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), title=None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)
    more_stopwords = {'com','http'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=max_font_size,
                          random_state=42,
                          mask=mask)
    wordcloud.generate(text)

    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask)
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
        plt.title(title, fontdict={'size': title_size}, verticalalignment='bottom')
    else:
        plt.imshow(wordcloud)
        plt.title(title, fontdict={'size': title_size, 'color': 'green'}, verticalalignment='bottom')

    plt.axis('off')
    plt.tight_layout()

#FOR GOOD TEXT
all_text=' '.join(good_sites['text'].tolist())
# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#FOR BAD TEXT
all_text=' '.join(bad_sites['text'].tolist())
# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#Vectorize
cv=CountVectorizer()
features=cv.fit_transform(df.text)
features[:5].toarray()

#SPLIT DATA
x_train,x_test,y_train,y_test= train_test_split(features,df.Label)

#MODEL TRAINING
l_model=LogisticRegression()
l_model.fit(x_train,y_train)
l_model.score(x_test,y_test)
l_model.score(x_train,y_train)

#CLASSIFICATION REPORT
# Predictions
y_pred = l_model.predict(x_test)

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=['Bad',"Good"]))

# Confusion Matrix
con_mat = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=["Predicted:Bad","Predicted:Good"],
    index=["Actual:Bad","Actual:Good"]
)

print("\nCONFUSION MATRIX")
plt.figure(figsize=(6,4))
sns.heatmap(con_mat, annot=True, fmt='d', cmap='YlGnBu')
plt.show()


#SAVE MODEL
pickle.dump(l_model,open('phishing.pk1','wb'))
pickle.dump(cv,open('vectorizer.pk1','wb'))
