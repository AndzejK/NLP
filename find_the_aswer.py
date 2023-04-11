import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer 
# getting another class
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

lemmatiser=WordNetLemmatizer()

text='Originally, vegetables were collected from the wild by hunter-gatherers. Vegetables are all plants. Vegetables can be eaten either raw or cooked. Vegetables can be eaten either raw or cooked.'
question='What are vegetables?'

def root_words(sentence):
    sentence_tokens=nltk.word_tokenize(sentence.lower())
    pos_tags=nltk.pos_tag(sentence_tokens)
    sentence_roots=[]
    for token,pos_tag in zip(sentence_tokens,pos_tags): 
        if pos_tag[1][0].lower() in ['n','v','a','r']:
            root=lemmatiser.lemmatize(token,pos_tag[1][0].lower())
            sentence_roots.append(root)
    return sentence_roots
sentence_tokens=nltk.sent_tokenize(text)
sentence_tokens.append(question)
print(sentence_tokens)
tv=TfidfVectorizer(tokenizer=root_words)
tf=tv.fit_transform(sentence_tokens)
print(tf.toarray())

# we compare question to all sentences in tf / to find their coefficients 
values=cosine_similarity(tf[-1],tf) # list in the list [[]]
print(values)
values_flat=values.flatten() # just the list - []
print(values_flat)
index=values.argsort()[0][-2]

coefficient=values_flat[-2]
print(coefficient)
if coefficient>0.3:
    print(sentence_tokens[index])