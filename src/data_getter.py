import os
import time
from io import BytesIO
import requests
from dotenv import load_dotenv

from PIL import Image
import pokemontcgsdk as poke

# Max number of acceptable fails to get try get a data set
max_fails = 5
# How long to wait in seconds if it fails getting a set before trying again
fail_retry_pause = 10

series = "xy"


def filter_type(card: poke.card.Card) -> bool:
    '''
    See if a card is a pokemon rather than a trainer or energy and test to make
    sure it is not a landscape big card too (called breaks).
    '''
    return card.supertype == "Pokémon" and "BREAK" not in card.subtypes

# Set our API key and create our client for getting data
load_dotenv()
API_KEY = os.getenv('API_KEY')
poke.RestClient.configure(API_KEY)

# Get all set IDS in a series
sets = poke.Set.where(q=f"series:{series}")
set_ids = [cardset.id for cardset in sets]

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
                    select="set,types,images,id,legalities,name,supertype,subtypes")
            break
        except poke.PokemonTcgException:
            print(f"Problem accessing {set_id} at attempt {fail_count}, trying again in 10 seconds")
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
        img.save(f"../data/{typ}/{card_id}.png")
    print(f"Images successfully cropped from {set_id}\n")
