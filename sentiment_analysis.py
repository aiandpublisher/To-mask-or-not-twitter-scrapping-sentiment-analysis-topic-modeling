
# first, we import the relevant modules from the NLTK library
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv('change_to_your_path_to_the_dataset.csv')


#replace URL of a text
def clean_data(df):

    df['text'] = df['text'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')

clean_data(df)
print(df['text']);


# next, we initialize VADER so we can use it within our Python script
vader = SentimentIntensityAnalyzer()
df['scores'] = df['text'].apply(lambda text: vader.polarity_scores(text))
df.head()


df['compound'] = df['scores'].apply(lambda score_dict:score_dict['compound'])
df.head()


df.to_csv('change_to_your_path.csv') 

# r is used to visualize the sentiment scores, check file sentiment_visualization

