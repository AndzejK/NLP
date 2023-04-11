from speech_recognition import Recognizer,AudioFile
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
recogniser=Recognizer()

with AudioFile('Work.wav') as audio_file:
    audio=recogniser.record(audio_file)

text=recogniser.recognize_google(audio)
print(text)
analyser=SentimentIntensityAnalyzer()
scores=analyser.polarity_scores(text)
print(scores)
if scores['compound']>0:
    print('Positive vibes!')
else:
    print('Negative vibes!')