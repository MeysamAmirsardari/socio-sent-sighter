import csv
import tweepy

# Autenticacion

consumer_key = "<>"  # Example ---> consumer_key = "asdad21424asdad12314"
consumer_secret = "<>"  # Example ---> consumer_secret = "asdad21424asdad12314"
access_token = "<>"  # Example ---> access_token="asdad21424asdad12314"
access_token_secret = "<>"  # Example ---> access_token_secret="asdad21424asdad12314"

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

# Prueba de credenciales
# print(api.verify_credentials().name)
# api.update_status(status='Updating using OAuth authentication via Tweepy!')

# Open/Create a file to append data
csvFile = open('ua.csv', 'a', encoding='utf-8', errors='replace')
# Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search_full_archive,
                           label='<>',  # Example ----> lavel ="developerenvironment"
                           query='#YourHasgtag place:"Your city"',
                           fromDate="YYYYMMDDHHMM",  # Example ----> fromDate="202003270000"
                           toDate="YYYYMMDDHHMM",  # Example ----> toDate="202003270000"
                           maxResults=100).items():
    csvWriter.writerow(
        ["URL: " + "https://twitter.com/user/status/" + str(tweet.id),
         " ---- Fecha de tweet: " + str(tweet.created_at),
         " ---- Ubicacion: " + str(tweet.place.name),
         " ---- Ubicacion del perfil: " + str(tweet.user.location),
         " ---- Nombre de Usuario: " + str(tweet.user.screen_name)
         + " ---- Cuerpo del tweet:  ", tweet.text])
    print("URL: " + "https://twitter.com/user/status/" + str(tweet.id),
          " ---- Fecha de tweet: " + str(tweet.created_at),
          " ---- Ubicacion: " + str(tweet.place.name),
          " ---- Ubicacion del perfil: " + str(tweet.user.location),
          " ---- Nombre de Usuario: " + str(tweet.user.screen_name)
          + " ---- Cuerpo del tweet:  ", tweet.text.encode('utf-8'))

csvFile.close()

# On vsCode
# ---- Just press "Run code" to generate the csv file - you'll find out that not all results can be printed due encodings problems (I think)
# ---- Just press "Run python file" I recommend dividing the search to one day at a time so that the results / tweets are not cut off in the terminal,
# or that you go to the settings and extend the number of lines that can be printed in the terminal so you can see all the tweets without any problems.