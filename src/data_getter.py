import os
import time
from io import BytesIO
from dotenv import load_dotenv

import requests
from PIL import Image
import pokemontcgsdk as poke

START_TIME = time.time()

def filter_type(card: poke.card.Card) -> bool:
    '''
    See if a card is a pokémon rather than a trainer or energy and test to make
    sure it is not a landscape big card too (called breaks).
    '''
    return card.supertype == "Pokémon" and "BREAK" not in card.subtypes

# Max number of acceptable fails to get try get a data set
max_fails = 5
# How long to wait in seconds if it fails getting a set before trying again
fail_retry_pause = 10

series_set = set(map(lambda s: s.series, poke.Set.all()))
print("All series:", ", ".join(series_set))

# Set our API key and create our client for getting data
load_dotenv()
API_KEY = os.getenv('API_KEY')
poke.RestClient.configure(API_KEY)

# Get all set IDS in all series
# Need the double quotes to deal with http request stuff not liking spaces in names
sets_per_series = [poke.Set.where(q=f"series:\"{series}\"") for series in series_set]
set_ids = [cardset.id for sets_in_one_series in sets_per_series for cardset in sets_in_one_series]

try:
    os.mkdir("../data")
except FileExistsError:
    pass

for set_id in set_ids:
    print(f"Getting cards from {set_id}")
    # If it fails, the else statement will execute
    for fail_count in range(max_fails):
        try:
            cards = poke.Card.where(q=f"set.id:{set_id}",\
                    select="set,types,images,id,legalities,name,supertype,subtypes,number")
            break
        except poke.PokemonTcgException:
            print(f"Problem accessing {set_id} at attempt {fail_count + 1}, trying again in 10 seconds")
            time.sleep(fail_retry_pause)
    else:
        print(f"Problem accessing {set_id}, giving up :(")
        continue
    print(f"Data successfully obtained from {set_id}")
    
    # Assume cards are all monotype pokemon and take the first type
    images = [(card.images.small, card.types[0], card.id) for card in filter(filter_type, cards)]

    for (url, typ, card_id) in images:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        # Chop out all of card picture that is not the picture bit
        # This means that full art cards are overcropped :(
        img = img.crop((21, 38, img.size[0]-21, img.size[1]-173))
        if not os.path.isdir(f"../data/{typ}"):
            os.mkdir(f"../data/{typ}")
        # ? cannot be ina file name, so replace it with url equivalent
        if "?" in card_id:
            card_id = card_id.replace("?", "%3F")
        img.save(f"../data/{typ}/{card_id}.png")
    print(f"All images successfully cropped from {set_id}\nCurrent total running time: {(time.time()-START_TIME) / 60} minutes\n")
