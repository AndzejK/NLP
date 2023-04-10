"""
 NLP - it is more natural way of searching for words, 
 basically it's trying to find the meaning rather than just bunch of strings.
 x = 'was'
 y = 'is'
 x = y # since they both come from verb "be" and that's where NLP can by handy!
 """
import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer # WordNetLemmatizer is class
x = 'is'
y = 'was '
x=y
lemmatiser=WordNetLemmatizer()
root_x=lemmatiser.lemmatize(x,"v")
root_y=lemmatiser.lemmatize(y,"v")
print(root_x==root_y)