import tweepy
import pandas as pd
import json

from scipy.misc import imread
import csv

client = tweepy.Client("add_your_tokens_here") #prerequisite: apply for the access to the API V2


results=[]
hash_tag = '(#WearAMask OR #MaskUp OR #Masks4All OR #FaceMasks OR #Masks OR #Mask OR #Masker OR #WearMask OR #MedicalFaceMask OR #VirusMask OR #HandMadeMask OR #SurgicalMask OR #MaskOn OR #TakeOffYourMask OR #MaskOff OR #NoMasks OR #UnmaskOurChildren OR #MaskBurning OR #TakeOffTheMask OR #NoMask OR #MasksDontWork OR #NoMaskOnMe OR #NoMasksOnMe or #NoMaskSelfie OR #BurnYourMaskChallenge OR #NoMasksEVER OR #NoMoreMasks OR #Masksoff OR #MasksMakeMeSweaty OR #NeverMasker OR #SheepWearMasks OR #MasksAreMurderingMe OR #StopForcingMasksOnMe OR #Maskhoax OR #Maskshoax OR #IWillNotWearAMask or #RefuseMask or #FaceFreedom or #StopMasking or #MomsAgainstMasks or #MaskingChildrenIsChildAbuse or #MasksAreForSheep or #StopWearingMask or #StopWearingTheDamnMasks or #MasksDontMatter) (place_country:GB lang:en)' #A set of hashtags about face masks
# change the country to CN for retrieving tweets in China
new_search = hash_tag + '-is:retweet'#exclude retweet


for tweet in tweepy.Paginator(client.search_all_tweets,new_search,start_time='2020-01-23T01:00:00Z',end_time='2021-7-19T01:00:00Z',tweet_fields=['created_at'], max_results=500).flatten(limit=100000):
	results.append(tweet).
# searching for tweets using hashtag


def tweets_df(results): # retrieve tweet text and creation time
	id_list = [tweet.id for tweet in results]
	data_set = pd.DataFrame(id_list, columns = ["id"])
	data_set["text"] = [tweet.text for tweet in results]
	data_set["created_at"] = [tweet.created_at for tweet in results]
	return data_set

data_set = tweets_df(results)
 

data_set.to_csv('change_to_your_own_path.csv')


