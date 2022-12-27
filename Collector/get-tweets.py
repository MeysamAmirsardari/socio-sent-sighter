import csv
import tweepy

# Autenticacion
consumer_key = "<>"
consumer_secret = "<>"
access_token = "<>"
access_token_secret = "<>"

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

cursor = tweepy.Cursor(api.search_full_archive,
                       label='<>',
                       query='#Karaj OR #کرج OR کرج OR entity:"Karaj" OR entity:"کرج" -is:retweet lang:fa -is:reply ',
                       fromDate="202209170000",
                       toDate="202210170000",
                       maxResults=1000).items()

csvFile = open('URLs_Karaj.csv', 'a', encoding='utf-8', errors='replace')
csvWriter = csv.writer(csvFile)

for tweet in cursor:
    csvWriter.writerow("https://twitter.com/" + str(tweet.user) + "/status/" + str(tweet.id))

csvFile.close()

csvFile = open('Content_Karaj.csv', 'a', encoding='utf-8', errors='replace')
csvWriter = csv.writer(csvFile)

for tweet in cursor:
    csvWriter.writerow(tweet.text)

csvFile.close()

csvFile = open('Meta_Karaj.csv', 'a', encoding='utf-8', errors='replace')
csvWriter = csv.writer(csvFile)

for tweet in cursor:
    csvWriter.writerow(
        [" ---- Created at: " + str(tweet.created_at),
         " ---- Location: " + str(tweet.place.name),
         " ---- User.Location: " + str(tweet.user.location),
         " ---- User.Name: " + str(tweet.user.screen_name)])

csvFile.close()
