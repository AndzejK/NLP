import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

analyser = SentimentIntensityAnalyzer()

txt_1="Hey cunt, how are you doing?"

txt_1_res=analyser.polarity_scores(txt_1)
print(txt_1_res)
