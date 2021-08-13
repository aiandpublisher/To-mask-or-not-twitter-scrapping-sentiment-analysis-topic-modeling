
# import the relevant modules from the NLTK library
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv('change_to_your_path_to_the_dataset.csv')

# add new score-specified words to the vader lexicon
new_words = {
    'wearamask': 5.0,
    'maskup': 5.0,
    'masks4all': 5.0,
    'wearmask':5.0,
    'maskon':5.0,
    'takeoffyourmask':-5.0,
    'maskoff':-5.0,
    'nomasks':-5.0,
    'unmaskourchildren':-5.0,
    'maskburning':-5.0,
    'takeoffthemask':-5.0,
    'nomask':-5.0,
    'masksdontwork':-5.0,
    'nomaskonme':-5.0,
    'nomasksonme':-5.0,
    'nomaskselfie':-5.0,
    'burnyourmaskchallenge':-5.0,
    'nomasksever':-5.0,
    'nomoremasks':-5.0,
    'masksoff':-5.0,
    'masksmakemesweaty':-5.0,
    'nevermasker':-5.0,
    'sheepwearmasks':-5.0,
    'masksaremurderingme':-5.0,
    'stopforcingmasksonme':-5.0,
    'maskhoax':-5.0,
    'maskshoax':-5.0,
    'iwillnotwearamask':-5.0,
    'refusemask':-5.0,
    'facefreedom':-5.0,
    'stopmasking':-5.0,
    'momsagainstmasks':-5.0,
    'maskingchildrenischildabuse':-5.0,
    'masksareforsheep':-5.0,
    'stopwearingmask':-5.0,
    'stopwearingthedamnmasks':-5.0,
    'masksdontmatter':-5.0
}


#clean texts

def clean_data(df):
    df['texts'] = df['text'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ') #remove hyperlinks
    df['texts'] = df['texts'].str.replace('\S*@\S*\s?', '') # remove emails
    df['texts'] = df['texts'].str.replace('\s+', ' ')  # remove newline chars
    df['texts'] = df['texts'].str.replace("\'", ' ') # remove single quotes
    df['texts'] = df['texts'].str.replace('@[\w]+','') #remove user names
    df['texts'] = df['texts'].str.replace('#','') #remove hashtag symbols
clean_data(df)

   

# initialize VADER to use it within the Python script
vader = SentimentIntensityAnalyzer()
# update the lexicon
vader.lexicon.update(new_words)
# test
text='Monitoring air quality at one of the final events of the Events Research Programme with a fantastic team! My mask doesn t bother me and it will stay on where possible indoors even after Freedom Day, because many people are not yet vaccinated. WearAMask '
print(vader.polarity_scores(text))


# calculate the sentiment polarities 
df['scores'] = df['text'].apply(lambda text: vader.polarity_scores(text))
df.head()
# calculate the compound score
df['compound'] = df['scores'].apply(lambda score_dict:score_dict['compound'])
df.head()



df.to_csv('change_to_your_path.csv') 
# r is used to visualize the sentiment scores, check file sentiment_visualization

