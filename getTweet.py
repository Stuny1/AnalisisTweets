import os
import tweepy as tw
import pandas as pd

consumer_key= 'Key'
consumer_secret= 'secret'
access_token= 'key'
access_token_secret= 'secret'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

search_words = "b"
date_since = "2020-11-16"

file= open('randomTweet.tsv','a+',encoding="utf-8")
file.write('id\ttweet\tlabel\n')


tweets = tw.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since).items(20000)



for tweet in tweets:
    print(tweet.text)
    file.write('id\t'+str(tweet.text).replace('\n',' ')+'\t'+'0\n')


