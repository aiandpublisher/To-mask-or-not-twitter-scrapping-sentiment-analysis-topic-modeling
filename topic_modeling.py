import sys
# !{sys.executable} -m spacy download en
# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
# Run in python console
import re
import numpy as np
import pandas as pd
from pprint import pprint
# spacy for lemmatization
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# customize stop word list
stop_words.extend(['from', 'subject', 're', 'especially','kinda','absolutely','totally','quite','fully','--','——','ー','bit','quite', 'actually','edu', 'would', 'could', 'amp','&amp', 'do','many', 'some', 'nice', 'rather', 'easy', 'easily', 'lot', 'seem', 'even', 'right', 'line', 'even', 'also', 'may', 'other','be','still','go','get','let','one','almost','-','--','s','much','point','indeed','otherwise','do','surely','right','lot','ever','often','mainly','obviously','even','hashtag','enough','exactly','however','exact','also','back','today','tonight','yet','extra','else','ve','really','instead','long','long','sure','already','sure','always','only','www'])
stop_words.extend(['maybe','truly'])
stop_words.extend(['tweet','a','mask','masks','wear','wearing','face','weared','put','facemask','facemasks','cover','facecoverings','facecovering','covers','covered','covering'])
stop_words.extend(['know','see','say','take','just','last','tomorrow','today','tonight','rd','gt','nt','twitter','morning','first','last'])
stop_words.extend(['twat','fuck','cunt','wanker','arse','ass','fucker','haha','sewarty'])

stop_words2 = [e for e in stop_words if e not in ('he', 'she','for','his','her','me','you','his','hers','against','myselves','yourselves','covidー','ff')]

%matplotlib inline
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

df = pd.read_csv('to_your_path.csv') 
df.dropna(axis='columns', inplace=True)
df.columns
df.drop_duplicates(inplace=True, subset="text") #remove duplicates

#manually split the hashtags for cleaner results, tokenize the texts
def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ',sent) #remove hyperlinks
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = re.sub(r'washyourhand', 'wash your hand', sent,flags=re.IGNORECASE)
        sent = re.sub(r'protectthenhs', 'protect the nhs', sent,flags=re.IGNORECASE)
        sent = re.sub(r'stayhome', 'stay home', sent,flags=re.IGNORECASE)
        sent = re.sub(r'borisjohnson', 'Boris Johnson', sent,flags=re.IGNORECASE)
        sent = re.sub(r'stayathome','stay at home',sent,flags=re.IGNORECASE)
        sent = re.sub(r'iprotectyouyouprotectme','i protect you you protect me',sent,flags=re.IGNORECASE)
        sent = re.sub(r'protecteachother','protect each other',sent,flags=re.IGNORECASE)
        sent = re.sub(r'staysafe','stay safe',sent,flags=re.IGNORECASE)
        sent = re.sub(r'stayhealthy','stay healthy',sent,flags=re.IGNORECASE)
        sent = re.sub(r'staypositive','stay positive',sent,flags=re.IGNORECASE)
        sent = re.sub(r'SocialDistancing','social distancing',sent,flags=re.IGNORECASE)
        sent = re.sub(r'#wearamask','wear a mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'wearamask','wear a mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'wedontconsent','we dont consent',sent,flags=re.IGNORECASE)
        sent = re.sub(r'novaccine','no vaccine',sent,flags=re.IGNORECASE)
        sent = re.sub(r'getvaccinate','get vaccinate',sent,flags=re.IGNORECASE)
        sent = re.sub(r'homesavelive','home save live',sent,flags=re.IGNORECASE)
        sent = re.sub(r'endthelockdown','end the lockdown',sent,flags=re.IGNORECASE)
        sent = re.sub(r'tunbridgewell','Tunbridge Well',sent,flags=re.IGNORECASE)
        sent = re.sub(r'coveryourface','cover your face',sent,flags=re.IGNORECASE)
        sent = re.sub(r'masksavealife','mask save a life',sent,flags=re.IGNORECASE)
        sent = re.sub(r'socialdistance','social distance',sent,flags=re.IGNORECASE)
        sent = re.sub(r'covidvaccine','covid vaccine',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nolockdown','no lockdown',sent,flags=re.IGNORECASE)
        sent = re.sub(r'secondwave','second wave',sent,flags=re.IGNORECASE)
        sent = re.sub(r'besafeoutthere','be save out there',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nomoremasks','no more masks',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nomaskonme','no mask on me',sent,flags=re.IGNORECASE)
        sent = re.sub(r'skincareroutine','skincare routine',sent,flags=re.IGNORECASE)
        sent = re.sub(r'coronavaccine','corona vaccine',sent,flags=re.IGNORECASE)
        sent = re.sub(r'johnsonmustgo','johnson must go',sent,flags=re.IGNORECASE)
        sent = re.sub(r'handsfacespace','hands face space',sent,flags=re.IGNORECASE)
        sent = re.sub(r'publichealth','public health',sent,flags=re.IGNORECASE)
        sent = re.sub(r'covidemergency','covid emergency',sent,flags=re.IGNORECASE)
        sent = re.sub(r'herdimmunity','herd immunity',sent,flags=re.IGNORECASE)
        sent = re.sub(r'fridaymorning','Friday morning',sent,flags=re.IGNORECASE)
        sent = re.sub(r'brexitshamble','brexit shamble',sent,flags=re.IGNORECASE)
        sent = re.sub(r'coviduk','covid uk',sent,flags=re.IGNORECASE)
        sent = re.sub(r'lockdownuk','lockdown uk',sent,flags=re.IGNORECASE)
        sent = re.sub(r'unmaskourchildren','unmask our children',sent,flags=re.IGNORECASE)
        sent = re.sub(r'wearadamnmask','wear a damn mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskup','mask up',sent,flags=re.IGNORECASE)
        sent = re.sub(r'facemask','face mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'facecovering','face covering',sent,flags=re.IGNORECASE)
        sent = re.sub(r'takeoffyourmask','take off your mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'wearadamnmask','wear a damn mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskoff','mask off',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nomask','no mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskoff','mask off',sent,flags=re.IGNORECASE)
        sent = re.sub(r'masks4all','masks for all',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nomasks','no masks',sent,flags=re.IGNORECASE)
        sent = re.sub(r'facemasks','face masks',sent,flags=re.IGNORECASE)
        sent = re.sub(r'wearmask','wearmask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'medicalfacemask','madical face mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'virusmask','virus mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'handmademask','handmade mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskburning','mask burning',sent,flags=re.IGNORECASE)
        sent = re.sub(r'surgicalmask','surgical mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskon','mask on',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nomask','no mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'masksdontwork','masks dont work',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskmaker','mask maker',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nomaskonme','no mask on me',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskfashion','mask fashion',sent,flags=re.IGNORECASE)
        sent = re.sub(r'fashionstyle','fashion style',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nomaskselfie','no mask selfie',sent,flags=re.IGNORECASE)
        sent = re.sub(r'burnyourmaskchallenge','burn your mask challenge',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nomasksever','no masks ever',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nomoremasks','no more masks',sent,flags=re.IGNORECASE)
        sent = re.sub(r'fridayfeeling','friday feeling',sent,flags=re.IGNORECASE)
        sent = re.sub(r'savetheplanet','save the planet',sent,flags=re.IGNORECASE)
        sent = re.sub(r'masksoff','masks off',sent,flags=re.IGNORECASE)
        sent = re.sub(r'wearthemask','wear the mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'masksmakemesweaty','masks make me sweaty',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nevermasker','never masker',sent,flags=re.IGNORECASE)
        sent = re.sub(r'sheepwearmasks','sheep wear masks',sent,flags=re.IGNORECASE)
        sent = re.sub(r'masksaremurderingme','masks are murdering me',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskhoax','mask hoax',sent,flags=re.IGNORECASE)
        sent = re.sub(r'stopforcingmasksonme','stop forcing masks on me',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskshoax','masks hoax',sent,flags=re.IGNORECASE)
        sent = re.sub(r'iwillnotwearamask','i will not wear a mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'refusemask','refuse mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'facefreedom','face freedom',sent,flags=re.IGNORECASE)
        sent = re.sub(r'borisisthelier','boris is the lier',sent,flags=re.IGNORECASE)
        sent = re.sub(r'stopmasking','stop masking',sent,flags=re.IGNORECASE)
        sent = re.sub(r'momsagainstmasks','moms against masks',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskingchildrenischildabuse','masking children is child abuse',sent,flags=re.IGNORECASE)
        sent = re.sub(r'masksareforsheep','masks are for sheep',sent,flags=re.IGNORECASE)
        sent = re.sub(r'stopwearingmask','stop wearing mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'stopwearingthedamnmasks','stop wearing the damn masks',sent,flags=re.IGNORECASE)
        sent = re.sub(r'masksdontmatter','masks dont matter',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nomorelockdown','no more lockdown',sent,flags=re.IGNORECASE)
        sent = re.sub(r'imdone','I am done',sent,flags=re.IGNORECASE)
        sent = re.sub(r'shutschool','shut school',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskmoaner','mask moaner',sent,flags=re.IGNORECASE)
        sent = re.sub(r'forhim','for him',sent,flags=re.IGNORECASE)
        sent = re.sub(r'forher','for her',sent,flags=re.IGNORECASE)
        sent = re.sub(r' ff', ' follow Friday', sent,flags=re.IGNORECASE)
        sent = re.sub(r' dm', ' direct message', sent,flags=re.IGNORECASE)
        sent = re.sub(r'kbf', 'keep Britain Free', sent,flags=re.IGNORECASE)
        sent = re.sub(r' remb ', ' remember ', sent,flags=re.IGNORECASE)
        sent = re.sub(r'dontbeaspreader','do not be a spreader',sent,flags=re.IGNORECASE)
        sent = re.sub(r'streetwearmask','street wear mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'deletetheapp','delete the app',sent,flags=re.IGNORECASE)
        sent = re.sub(r'stopcomplying','stop complying',sent,flags=re.IGNORECASE)
        sent = re.sub(r'enoughisenough','enough is enough',sent,flags=re.IGNORECASE)
        sent = re.sub(r'freedomday','freedom day',sent,flags=re.IGNORECASE)
        sent = re.sub(r'boycottnhsapp','boycott nhs app',sent,flags=re.IGNORECASE)
        sent = re.sub(r'WeKnowYouAreLying','we know you are lying ',sent,flags=re.IGNORECASE)
        sent = re.sub(r'realdonaldtrump','real donald trump',sent,flags=re.IGNORECASE)
        sent = re.sub(r'SCAMDEMIC2021','scamdemic 2021',sent,flags=re.IGNORECASE)
        sent = re.sub(r'endlockdown','end lockdown',sent,flags=re.IGNORECASE)
        sent = re.sub(r'nonewnormal','no new nomal',sent,flags=re.IGNORECASE)
        sent = re.sub(r'takeoffthemask','take off the mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'malemodel','male model',sent,flags=re.IGNORECASE)
        sent = re.sub(r'neempowder','neem powder',sent,flags=re.IGNORECASE)
        sent = re.sub(r'wedonotconsent','we do not consent',sent,flags=re.IGNORECASE)
        sent = re.sub(r"is'nt",'is not',sent,flags=re.IGNORECASE)
        sent = re.sub(r"arnt",'are not',sent,flags=re.IGNORECASE)
        sent = re.sub(r'genderequality','gender equality',sent,flags=re.IGNORECASE)   
        sent = re.sub(r'noface','no face',sent,flags=re.IGNORECASE)    
        sent = re.sub(r'stopthemadness','stop the madness',sent,flags=re.IGNORECASE)  
        sent = re.sub(r'notomask','no to mask',sent,flags=re.IGNORECASE)  
        sent = re.sub(r'the pm','the prime minister',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'co-op','coop',sent,flags=re.IGNORECASE)   
        sent = re.sub(r'johnsonout','johnson out',sent,flags=re.IGNORECASE)  
        sent = re.sub(r'nomuzzle','no muzzle',sent,flags=re.IGNORECASE)   
        sent = re.sub(r'dontgotothepub','dont go to the pub',sent,flags=re.IGNORECASE)   
        sent = re.sub(r'saynotolockdown','say no to lock down',sent,flags=re.IGNORECASE)  
        sent = re.sub(r'speakyourtruth','speak your truth',sent,flags=re.IGNORECASE)   
        sent = re.sub(r'tuesdaythought','Tuesday thought',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'brexitsacrime','brexits a crime',sent,flags=re.IGNORECASE)
        sent = re.sub(r'governmentsheep','government sheep',sent,flags=re.IGNORECASE)  
        sent = re.sub(r'covididiot','covid idiot',sent,flags=re.IGNORECASE)        
        sent = re.sub(r'vaccineswork','vaccines work',sent,flags=re.IGNORECASE)  
        sent = re.sub(r'wearmask','wear mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'covidchristma','covid christma',sent,flags=re.IGNORECASE)
        sent = re.sub(r'savelife','save life',sent,flags=re.IGNORECASE)     
        sent = re.sub(r'lol','laugh out loud',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'testchecktrace','test check trace',sent,flags=re.IGNORECASE)   
        sent = re.sub(r'protectother','protect other',sent,flags=re.IGNORECASE)   
        sent = re.sub(r'govt','government',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'masksformen','masks for men',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'masksforwomen','masks for women',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'publictransport','public transport',sent,flags=re.IGNORECASE)
        sent = re.sub(r'shopsreopening','shops reopening',sent,flags=re.IGNORECASE)
        sent = re.sub(r'keep safe','keep safe',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskssavealife','masks save life',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskofcotton','mask of cotton',sent,flags=re.IGNORECASE)
        sent = re.sub(r'doglover','dog lover',sent,flags=re.IGNORECASE)
        sent = re.sub(r'trickortreat','trick or treat',sent,flags=re.IGNORECASE)
        sent = re.sub(r'respectthevirus','respect the virus',sent,flags=re.IGNORECASE)
        sent = re.sub(r'mutualaid','mutual aid',sent,flags=re.IGNORECASE)
        sent = re.sub(r'fabricmask','fabrick mask',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'boho fashion','boho fashion',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'masksforall','masks for all',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'shopalone','shop alone',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'fabricface','fabric face',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'keepsafe','keep safe',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'buyface','buy face',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'protectivemask','protective mask',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'facecover','face cover',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'wearyourmask','wear your mask',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'stayalert','stay alert',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'savetheart','save the art',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'africanfashion','african fashion',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'vaccinnatedasap','vaccinated asap',sent,flags=re.IGNORECASE) 
        sent = re.sub(r'lepardprint','lepard print',sent,flags=re.IGNORECASE)
        sent = re.sub(r'lockdownlife','lockdown life',sent,flags=re.IGNORECASE)
        sent = re.sub(r'schoolclosure','school closure',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maidenheadtoday','maidenhead today',sent,flags=re.IGNORECASE)
        sent = re.sub(r'healthandsafety','health and safety',sent,flags=re.IGNORECASE)
        sent = re.sub(r'londonfashion','london fashion',sent,flags=re.IGNORECASE)
        sent = re.sub(r'colorfulmask','colorful mask',sent,flags=re.IGNORECASE)
        sent = re.sub(r'latexfashion','latex fashion',sent,flags=re.IGNORECASE)
        sent = re.sub(r'masksinclass','masks in class',sent,flags=re.IGNORECASE)
        sent = re.sub(r'faceprotection','face protection',sent,flags=re.IGNORECASE)
        sent = re.sub(r'latexgirl','latex girl',sent,flags=re.IGNORECASE)
        sent = re.sub(r'sackjohnson','sack johnson',sent,flags=re.IGNORECASE)
        sent = re.sub(r'thankyou','thank you',sent,flags=re.IGNORECASE)
        sent = re.sub(r'mondaythought','monday thought',sent,flags=re.IGNORECASE)
        sent = re.sub(r'savelive','save live',sent,flags=re.IGNORECASE)
        sent = re.sub(r'covidisnotover','covid is not over',sent,flags=re.IGNORECASE)
        sent = re.sub(r'spacefreshair','space fresh air',sent,flags=re.IGNORECASE)
        sent = re.sub(r'testtracktrace','test track trace',sent,flags=re.IGNORECASE)
        sent = re.sub(r'wednesdaywisdom','wednesday wisdom',sent,flags=re.IGNORECASE)
        sent = re.sub(r'monsonnews','monson news',sent,flags=re.IGNORECASE)
        sent = re.sub(r' pic ',' picture ',sent,flags=re.IGNORECASE)
        sent = re.sub(r'twitterblock','twitter block',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskssavelives','masks save lives',sent,flags=re.IGNORECASE)
        sent = re.sub(r'maskssave','masks save',sent,flags=re.IGNORECASE)
        sent = re.sub(r'trackandtrace','track and trace',sent,flags=re.IGNORECASE)
        sent = re.sub(r'backtoschool','back to school',sent,flags=re.IGNORECASE)
        sent = re.sub(r'eatouttohelp','eat out to help',sent,flags=re.IGNORECASE)
        sent = re.sub(r'madetoorder','made to order',sent,flags=re.IGNORECASE)
        sent = re.sub(r'massobservation','mass observation',sent,flags=re.IGNORECASE)
        sent = re.sub(r'contemporaryart','contemporary art',sent,flags=re.IGNORECASE)
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  



# Convert to list
data = df.text.values.tolist() #modify 'text' to the column name of the twitter data 
data_words_needs_clean = list(sent_to_words(data))
# data_words_neads_clean[:1] try this to check 


# reclean the duplicated tweets after hashtag processing,
def sortAndUniq(input):
  output = []
  for x in input:
    if x not in output:
      output.append(x)
  output.sort()
  return output


data_words=sortAndUniq(data_words_needs_clean) # only 13541 records were kept for topic modeling


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10, delimiter='_') # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], min_count=5, threshold=10, delimiter='_')  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


#remove stop words
def remove_stopwords(texts,stop):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]



# make sure this step is executed and NOT SKIPPED
# make sure this step is executed and NOT SKIPPED
# make sure this step is executed and NOT SKIPPED
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def process_words(texts, allowed_postags=['NOUN']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out


# Remove some new Stop Words from the result
data_words_nostops = remove_stopwords(data_words,stop_words2)

# Form Bigrams and trigrams
data_words_bigrams = make_bigrams(data_words_nostops)
data_words_trigrams = make_trigrams(data_words_bigrams)

# processed Text Data and only keep nouns
data_ready = process_words(data_words_bigrams, allowed_postags=['NOUN'])  

# Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]
print(corpus[:1])



#optimize topic models by coherence values
coherence_values = []

for num_topics in range(5,35,2): # for saving the data processing time this study used 2 as the step value for positive dataset and 0 for negative dataset
	print('Round: '+str(num_topics))
	Lda = gensim.models.ldamodel.LdaModel
	ldamodel = Lda(corpus, num_topics=num_topics,id2word=id2word,passes=200, iterations=2000, chunksize=20000, eval_every = None, random_state=0,alpha='auto',eta='auto')   	
	coherencemodel = CoherenceModel(
	   	model=ldamodel, texts=data_ready, dictionary=id2word, coherence='c_v'
	   	)
	coherence_values.append(coherencemodel.get_coherence())


coherence_values

#use r studio for plotting coherence values or use the following code
# model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=docs, start=2, limit=40, step=6)
# import matplotlib.pyplot as plt
# limit=35; start=5, step=2; 
# x = range(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()
	

	

# train lda model
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=12,id2word=id2word, passes=200, iterations=2000, chunksize=20000, eval_every = None, random_state=0,alpha='auto',eta='auto')
#Chunksize larger than the total number of tweets takes in all tweets in one go. 
#passes to 80; chosen by setting an argument 
#eval_every = 0 to reduce time. 
#alpha = 'auto' and eta = 'auto', which will automatically learn these parameters.




# measure coherence scores & perplexity score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_ready, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.


# visualizaton
topic_data =  pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds = 'mmds')

# Export the visualization as a html file.
pyLDAvis.save_html(topic_data, 'to_your_path.html')


# you could also arrange the topics using a table
all_topics = {}
num_terms = 15 # Adjust number of words to represent each topic
lambd = 0.60 # Adjust this accordingly based on tuning above
for i in range(1,18): #Adjust this to reflect number of topics chosen for final LDA model
    topic = topic_data.topic_info[topic_data.topic_info.Category == 'Topic'+str(i)].copy()
    topic['relevance'] = topic['loglift']*(1-lambd)+topic['logprob']*lambd
    all_topics['Topic '+str(i)] = topic.sort_values(by='relevance', ascending=False).Term[:num_terms].values

topicsplot=pd.DataFrame(all_topics).T
topicsplot.to_csv('to_your_path.csv')

