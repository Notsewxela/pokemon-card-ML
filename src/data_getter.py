import pokemontcgsdk as poke
import os
from dotenv import load_dotenv

from PIL import Image
import requests
from io import BytesIO


series = "xy"


def filter_type(card: poke.card.Card) -> bool:
    '''
    See if a card is a pokemon rather than a trainer or energy and test to make
    sure it is not a landscape big card too (called breaks).
    '''
    return card.supertype == "Pokémon" and "BREAK" not in card.subtypes

load_dotenv()
API_KEY = os.getenv('API_KEY')
poke.RestClient.configure(API_KEY)

sets = poke.Set.where(q=f"series:{series}")
set_ids = [cardset.id for cardset in sets]

try:
    os.mkdir("../data")
except FileExistsError:
    pass

for set_id in set_ids:
    print(set_id)
    cards = poke.Card.where(q=f"set.id:{set_id}", select="set,types,images,id,legalities,name,supertype,subtypes")
    # Assume cards are all monotype pokemon and take the first type
    images = [(card.images.small, card.types[0]) for card in filter(filter_type, cards)]

    for i, (url, typ) in enumerate(images):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        # Chop out all of card picture that is not the picture bit
        # This means that full art cards are overcropped :(
        img = img.crop((21, 38, img.size[0]-21, img.size[1]-173))
        img.save(f"../data/{set_id}-{typ}-{i+1}.png")