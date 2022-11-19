## Pokémon card type inference from card picture

This project loads a series of pokémon cards, and then will try to learn the types from the picture on the card. The intention is to pass the algorithm any image that looks like a pokemon and guess the type!

### Requirements

To run this code, you need:
- a `.env` file in the top level directory containing an `API_KEY` environment variable allowing access to the [Pokémon TCG IO API](https://dev.pokemontcg.io/).
- pokemontcgsdk python library
- tensorflow python library
- PILLOW python library

All the libraries are pip installable.
