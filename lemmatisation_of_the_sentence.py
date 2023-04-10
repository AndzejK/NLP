import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer 
lemmatiser=WordNetLemmatizer()
sentence='Tokenising this sentence but not the sentences  doing in order to learn some shit! '
# tokenisation is the method to work with sentences and each word is broken down into token
#result_tokens=nltk.word_tokenize(sentence.lower()) # now each word/token is listed/stored in the list
# looping through the list - result_tokens
# print(result_tokens)
# pos - Part Of Speech
#pos_tags=nltk.pos_tag(result_tokens)
#print(pos_tags)
def root_words(senten):
    result_tokens=nltk.word_tokenize(senten.lower())
    pos_tags=nltk.pos_tag(result_tokens)
    sentence_roots=[]
# instead of adding separatly/manually part of grammer we're gonna use tags nad automate it
    for token,pos_tag in zip(result_tokens,pos_tags): # iterate at the same time through to diff Lists but having the same number of items
        if pos_tag[1][0].lower() in ['n','v','a','r']:
            root=lemmatiser.lemmatize(token,pos_tag[1][0].lower()) # pos_tags[1][0].lower(), since it is a tuple that has two values and we're interested in the 2nd value and just first letter, telling us what part of gram it's
            sentence_roots.append(root)
        return sentence_roots
    # else:
    #     print(token)

print(root_words('Being a coder can open up a bunch of  doors , did you get that? I did'))