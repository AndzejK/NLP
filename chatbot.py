import nltk
from nltk.stem import WordNetLemmatizer 
# getting another class
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia

lemmatiser=WordNetLemmatizer()

text=wikipedia.page('Vegetables').content
print(text)

def root_words(sentence):
    sentence_tokens=nltk.word_tokenize(sentence.lower())
    pos_tags=nltk.pos_tag(sentence_tokens)
    sentence_roots=[]
    for token,pos_tag in zip(sentence_tokens,pos_tags): 
        if pos_tag[1][0].lower() in ['n','v','a','r']:
            root=lemmatiser.lemmatize(token,pos_tag[1][0].lower())
            sentence_roots.append(root)
    return sentence_roots

def process(text,question):
    sentence_tokens=nltk.sent_tokenize(text)
    sentence_tokens.append(question)
    tv=TfidfVectorizer(tokenizer=root_words)
    tf=tv.fit_transform(sentence_tokens)

    # we compare question to all sentences in tf / to find their coefficients 
    values=cosine_similarity(tf[-1],tf) # list in the list [[]]
    values_flat=values.flatten() # just the list - []
    index=values.argsort()[0][-2]
    coefficient=values_flat[-2]
    if coefficient>0.3:
        return sentence_tokens[index]

while True:
    question=input("What would you like to ask me?\n") 
    output=process(text=text,question=question)
    if output:
        print(output)
    elif question=='quit':
        break
    else:
        print("I have no clue what you're talking about, mate!")