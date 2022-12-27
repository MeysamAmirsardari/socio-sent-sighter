import csv
import tweepy


bearer_token = ""

client = tweepy.Client(bearer_token)

csvFile = open('URLs_Karaj.csv', 'a', encoding='utf-8', errors='replace')
csvReader = csv.reader(csvFile)

tweet_id = 1460323737035677698

response = client.get_retweeters(tweet_id, user_fields=["profile_image_url"])

for user in response.data:
    print(user.username, user.profile_image_url)