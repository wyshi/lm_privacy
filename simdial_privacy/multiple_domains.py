# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
'''
python multiple_domains.py --domain track_package --complexity mix --train_size 10000 --test_size 1000 --valid_size 1000 --save_dir output
'''
from simdial.domain import Domain, DomainSpec
from simdial.generator import Generator
from simdial import complexity
import string
import argparse
import os
import json
import pandas as pd
import numpy as np

class RestSpec(DomainSpec):
    name = "restaurant"
    greet = "Welcome to restaurant recommendation system."
    nlg_spec = {"loc": {"inform": ["I am at %s.", "%s.", "I'm interested in food at %s.", "At %s.", "In %s."],
                        "request": ["Which city are you interested in?", "Which place?"]},

                "food_pref": {"inform": ["I like %s food.", "%s food.", "%s restaurant.", "%s."],
                              "request": ["What kind of food do you like?", "What type of restaurant?"]},

                "open": {"inform": ["The restaurant is %s.", "It is %s right now."],
                         "request": ["Tell me if the restaurant is open.", "What's the hours?"],
                         "yn_question": {'open': ["Is the restaurant open?"],
                                         'closed': ["Is it closed?"]
                                         }},

                "parking": {"inform": ["The restaurant has %s.", "This place has %s."],
                            "request": ["What kind of parking does it have?.", "How easy is it to park?"],
                            "yn_question": {'street parking': ["Does it have street parking?"],
                                            "valet parking": ["Does it have valet parking?"]
                                            }},

                "price": {"inform": ["The restaurant serves %s food.", "The price is %s."],
                          "request": ["What's the average price?", "How expensive it is?"],
                          "yn_question": {'expensive': ["Is it expensive?"],
                                          'moderate': ["Does it have moderate price?"],
                                          'cheap': ["Is it cheap?"]
                                          }},

                "default": {"inform": ["Restaurant %s is a good choice."],
                            "request": ["I need a restaurant.",
                                        "I am looking for a restaurant.",
                                        "Recommend me a place to eat."]}
                }

    usr_slots = [("loc", "location city", ["Pittsburgh", "New York", "Boston", "Seattle",
                                           "Los Angeles", "San Francisco", "San Jose",
                                           "Philadelphia", "Washington DC", "Austin"]),
                 ("food_pref", "food preference", ["Thai", "Chinese", "Korean", "Japanese",
                                                   "American", "Italian", "Indian", "French",
                                                   "Greek", "Mexican", "Russian", "Hawaiian"])]

    sys_slots = [("open", "if it's open now", ["open", "closed"]),
                 ("price", "average price per person", ["cheap", "moderate", "expensive"]),
                 ("parking", "if it has parking", ["street parking", "valet parking", "no parking"])]

    db_size = 100


class RestStyleSpec(DomainSpec):
    name = "restaurant_style"
    greet = "Hello there. I know a lot about places to eat."
    nlg_spec = {"loc": {"inform": ["I am at %s.", "%s.", "I'm interested in food at %s.", "At %s.", "In %s."],
                        "request": ["Which area are you currently locating at?", "well, what is the place?"]},

                "food_pref": {"inform": ["I like %s food.", "%s food.", "%s restaurant.", "%s."],
                              "request": ["What cusine type are you interested", "What do you like to eat?"]},

                "open": {"inform": ["This wonderful place is %s.", "Currently, this place is %s."],
                         "request": ["Tell me if the restaurant is open.", "What's the hours?"],
                         "yn_question": {'open': ["Is the restaurant open?"],
                                         'closed': ["Is it closed?"]
                                         }},

                "parking": {"inform": ["The parking status is %s.", "For parking, it does have %s."],
                            "request": ["What kind of parking does it have?.", "How easy is it to park?"],
                            "yn_question": {'street parking': ["Does it have street parking?"],
                                            "valet parking": ["Does it have valet parking?"]
                                            }},

                "price": {"inform": ["This eating place provides %s food.", "Let me check that for you. The price is %s."],
                          "request": ["What's the average price?", "How expensive it is?"],
                          "yn_question": {'expensive': ["Is it expensive?"],
                                          'moderate': ["Does it have moderate price?"],
                                          'cheap': ["Is it cheap?"]
                                          }},

                "default": {"inform": ["Let me look up in my database. A good choice is %s."],
                            "request": ["I need a restaurant.",
                                        "I am looking for a restaurant.",
                                        "Recommend me a place to eat."]}
                }

    usr_slots = [("loc", "location city", ["Pittsburgh", "New York", "Boston", "Seattle",
                                           "Los Angeles", "San Francisco", "San Jose",
                                           "Philadelphia", "Washington DC", "Austin"]),
                 ("food_pref", "food preference", ["Thai", "Chinese", "Korean", "Japanese",
                                                   "American", "Italian", "Indian", "French",
                                                   "Greek", "Mexican", "Russian", "Hawaiian"])]

    sys_slots = [("open", "if it's open now", ["open", "closed"]),
                 ("price", "average price per person", ["cheap", "moderate", "expensive"]),
                 ("parking", "if it has parking", ["street parking", "valet parking", "no parking"])]

    db_size = 100


class RestPittSpec(DomainSpec):
    name = "rest_pitt"
    greet = "I am an expert about Pittsburgh restaurant."

    nlg_spec = {"loc": {"inform": ["I am at %s.", "%s.", "I'm interested in food at %s.", "At %s.", "In %s."],
                        "request": ["Which city are you interested in?", "Which place?"]},

                "food_pref": {"inform": ["I like %s food.", "%s food.", "%s restaurant.", "%s."],
                              "request": ["What kind of food do you like?", "What type of restaurant?"]},

                "open": {"inform": ["The restaurant is %s.", "It is %s right now."],
                         "request": ["Tell me if the restaurant is open.", "What's the hours?"],
                         "yn_question": {'open': ["Is the restaurant open?"],
                                         'closed': ["Is it closed?"]
                                         }},

                "parking": {"inform": ["The restaurant has %s.", "This place has %s."],
                            "request": ["What kind of parking does it have?.", "How easy is it to park?"],
                            "yn_question": {'street parking': ["Does it have street parking?"],
                                            "valet parking": ["Does it have valet parking?"]
                                            }},

                "price": {"inform": ["The restaurant serves %s food.", "The price is %s."],
                          "request": ["What's the average price?", "How expensive it is?"],
                          "yn_question": {'expensive': ["Is it expensive?"],
                                          'moderate': ["Does it have moderate price?"],
                                          'cheap': ["Is it cheap?"]
                                          }},

                "default": {"inform": ["Restaurant %s is a good choice."],
                            "request": ["I need a restaurant.",
                                        "I am looking for a restaurant.",
                                        "Recommend me a place to eat."]}
                }

    usr_slots = [("loc", "location city", ["Downtown", "CMU", "Forbes and Murray", "Craig",
                                           "Waterfront", "Airport", "U Pitt", "Mellon Park",
                                           "Lawrance", "Monroveil", "Shadyside", "Squrill Hill"]),
                 ("food_pref", "food preference", ["healthy", "fried", "panned", "steamed", "hot pot",
                                                   "grilled", "salad", "boiled", "raw", "stewed"])]

    sys_slots = [("open", "if it's open now", ["open", "going to start", "going to close", "closed"]),
                 ("price", "average price per person", ["cheap", "average", "fancy"]),
                 ("parking", "if it has parking", ["garage parking", "street parking", "no parking"])]

    db_size = 150


class BusSpec(DomainSpec):
    name = "bus"
    greet = "Ask me about bus information."

    nlg_spec = {"from_loc": {"inform": ["I am at %s.", "%s.", "Leaving from %s.", "At %s.", "Departure place is %s."],
                             "request": ["Where are you leaving from?", "What's the departure place?"]},

                "to_loc": {"inform": ["Going to %s.", "%s.", "Destination is %s.", "Go to %s.", "To %s"],
                           "request": ["Where are you going?", "Where do you want to take off?"]},

                "datetime": {"inform": ["At %s.", "%s.", "I am leaving on %s.", "Departure time is %s."],
                             "request": ["When are you going?", "What time do you need the bus?"]},

                "arrive_in": {"inform": ["The bus will arrive in %s minutes.", "Arrive in %s minutes.",
                                         "Will be here in %s minutes"],
                              "request": ["When will the bus arrive?", "How long do I need to wait?",
                                          "What's the estimated arrival time"],
                              "yn_question": {k: ["Is it a long wait?"] if k>15 else ["Will it be here shortly?"]
                                              for k in range(0, 30, 5)}},

                "duration": {"inform": ["It will take %s minutes.", "The ride is %s minutes long."],
                             "request": ["How long will it take?.", "How much tim will it take?"],
                              "yn_question": {k: ["Will it take long to get there?"] if k>30 else ["Is it a short trip?"]
                                              for k in range(0, 60, 5)}},

                "default": {"inform": ["Bus %s can take you there."],
                            "request": ["Look for bus information.",
                                        "I need a bus.",
                                        "Recommend me a bus to take."]}
                }

    usr_slots = [("from_loc", "departure place", ["Downtown", "CMU", "Forbes and Murray", "Craig",
                                                  "Waterfront", "Airport", "U Pitt", "Mellon Park",
                                                  "Lawrance", "Monroveil", "Shadyside", "Squrill Hill"]),
                 ("to_loc", "arrival place", ["Downtown", "CMU", "Forbes and Murray", "Craig",
                                              "Waterfront", "Airport", "U Pitt", "Mellon Park",
                                              "Lawrance", "Monroveil", "Shadyside", "Squrill Hill"]),
                 ("datetime", "leaving time", ["today", "tomorrow", "tonight", "this morning",
                                               "this afternoon"] + [str(t+1) for t in range(24)])
                 ]

    sys_slots = [("arrive_in", "how soon it arrives", [str(t) for t in range(0, 30, 5)]),
                 ("duration", "how long it takes", [str(t) for t in range(0, 60, 5)])
                 ]

    db_size = 150


class WeatherSpec(DomainSpec):
    name = "weather"
    greet = "Weather bot is here."

    nlg_spec = {"loc": {"inform": ["I am at %s.", "%s.", "Weather at %s.", "At %s.", "In %s."],
                        "request": ["Which city are you interested in?", "Which place?"]},

                "datetime": {"inform": ["Weather %s", "%s.", "I am interested in %s."],
                             "request": ["What time's weather?", "What date are you interested?"]},

                "temperature": {"inform": ["The temperature will be %s.", "The temperature that time will be %s."],
                                "request": ["What's the temperature?", "What will be the temperature?"]},

                "weather_type": {"inform": ["The weather will be %s.", "The weather type will be %s."],
                                 "request": ["What's the weather type?.", "What will be the weather like"],
                                 "yn_question": {k: ["Is it going to be %s?" % k] for k in
                                                 ["raining", "snowing", "windy", "sunny", "foggy", "cloudy"]}
                                 },

                "default": {"inform": ["Your weather report %s is here."],
                            "request": ["What's the weather?.",
                                        "What will the weather be?"]}
                }

    usr_slots = [("loc", "location city", ["Pittsburgh", "New York", "Boston", "Seattle",
                                           "Los Angeles", "San Francisco", "San Jose",
                                           "Philadelphia", "Washington DC", "Austin"]),
                 ("datetime", "which time's weather?", ["today", "tomorrow", "tonight", "this morning",
                                                        "the day after tomorrow", "this weekend"])]

    sys_slots = [("temperature", "the temperature", [str(t) for t in range(20, 40, 2)]),
                 ("weather_type", "the type", ["raining", "snowing", "windy", "sunny", "foggy", "cloudy"])]

    db_size = 40


class MovieSpec(DomainSpec):
    name = "movie"
    greet = "Want to know about movies?"

    nlg_spec = {"genre": {"inform": ["I like %s movies.", "%s.", "I love %s ones.", "%s movies."],
                          "request": ["What genre do you like?", "Which type of movie?"]},

                "years": {"inform": ["Movies in %s", "In %s."],
                          "request": ["What's the time period?", "Movie in what years?"]},

                "country": {"inform": ["Movie from %s", "%s.", "From %s."],
                            "request": ["Which country's movie?", "Movie from what country?"]},

                "rating": {"inform": ["This movie has a rating of %s.", "The rating is %s."],
                           "request": ["What's the rating?", "How people rate this movie?"],
                           "yn_question": {"5": ["Does it have a perfect rating?"],
                                           "4": ["Does it have a rating of 4/5?"],
                                           "1": ["Does it have a very bad rating?"]}
                           },

                "company": {"inform": ["It's made by %s.", "The movie is from %s."],
                            "request": ["Which company produced this movie?.", "Which company?"],
                            "yn_question": {k: ["Is this movie from %s?" % k] for k in
                                            ["20th Century Fox", "Sony", "MGM", "Walt Disney", "Universal"]}
                            },

                "director": {"inform": ["The director is %s.", "It's director by %s."],
                             "request": ["Who is the director?.", "Who directed it?"],
                             "yn_question": {k: ["Is it directed by %s?" % k] for k in
                                             list(string.ascii_uppercase)}
                             },

                "default": {"inform": ["Movie %s is a good choice."],
                            "request": ["Recommend a movie.",
                                        "Give me some good suggestions about movies.",
                                        "What should I watch now"]}
                }

    usr_slots = [("genre", "type of movie", ["Action", "Sci-Fi", "Comedy", "Crime",
                                             "Sport", "Documentary", "Drama",
                                             "Family", "Horror", "War", "Music", "Fantasy", "Romance", "Western"]),

                 ("years", "when", ["60s", "70s", "80s", "90s", "2000-2010", "2010-present"]),

                 ("country", "where ", ["USA", "France", "China", "Korea",
                                        "Japan", "Germany", "Mexico", "Russia", "Thailand"])
                 ]

    sys_slots = [("rating", "user rating", [str(t) for t in range(5)]),
                 ("company", "the production company", ["20th Century Fox", "Sony", "MGM", "Walt Disney", "Universal"]),
                 ("director", "the director's name", list(string.ascii_uppercase))
                 ]

    db_size = 200
    
# New spec created for the privacy project
class TrackPackageSpec(DomainSpec):
    def __init__(self, one_token, num_info_ask):
        if one_token.lower() in ('yes', 'true', 't', 'y', '1'):
            self.one_token_private_info = True
        else:
            self.one_token_private_info = False
        self.num_info_ask = num_info_ask
    
        self.name = "track_package"
        self.greet = "Hello, I am with customer support bot."

        self.nlg_spec = {"name": {"inform": ["I am %s.", "%s.", "Sure, %s.", "Yes, %s.", "%s", "Yep - I'm %s.", "The name's %s."],
                            "request": ["May I have your full name please?", "Can you verify your full name so I can look that up?", "Please provide your full name"]},

                    "phone": {"inform": ["Phone number is %s", "%s.", "You can reach me at %s.", "%s is my number.", "my number is %."],
                                "request": ["Ok, let me get your phone number really quick.", "Verify your phone number please."]},

                    "address": {"inform": ["My address is %s.", "%s.", "Ok, it is %s.", "Yea sure, %s.", "Shipping address is %s."],
                                    "request": ["We will need the shipping address as well.", "Could you please confirm your shipping address?"]},

                    "shipment": {"inform": ["Your package has been delivered.", "Your package will arrive %s.", "%s.", "%s is the arrival date", "You package will be delivered %s.", "Your package will arrive %s."],
                                "request": ["When can I receive my package", "When will it be delivered?", "What is the delivery date?", "When will the package arrive?", "When will it arrive"],
                                "yn_question": {'status': ["Is it shipped?"],
                                            'deliver': ["Is it delivered?"]
                                            }},
                    "order_number": {"inform": ["Sure, it is %s", "%s", "It's %s.", "Yes, %s.", "My order number is %s."],
                                "request": ["Could you please also provide your order number?", "Verify your order number please.","Can you provide the order number?"]},

                    "default": {"inform": ["The tracking number of your package is %s.", "You can track your package using your tracking number, which is %s.", "Track your order using your tracking number, %s."],
                                "request": ["Where is my package?",
                                            "Could you please help me track my package?",
                                            "I placed an order but I don't know if it has been shipped."] + ["I ordered a %s several days ago but I can't track it." % k for k in
                                                ["lipstick", "mobile phone", "pot", "floor lamp", "chair"]]}
                    }
        rand_names, rand_addresses, rand_phone_numbers, rand_card_numbers, rand_order_numbers = read_rand_entity_db("database/database_500.csv")

        self.usr_slots = [("name", "customer name", rand_names),
                    ("phone", "customer phone number", rand_phone_numbers),
                    ("order_number", "customer order number", rand_order_numbers),
                    ("address", "customer shipping address",rand_addresses)]

        self.sys_slots = [("shipment", "expected shipment date", ["today", "tomorrow", "tonight", "this morning",
                                                            "the day after tomorrow", "this weekend"])]

        self.db_size = 200

def read_rand_entity_db(path):
    df = pd.read_csv(path)

    rand_names = df["name"].tolist()
    rand_addresses = df["address"].tolist()
    rand_phone_numbers = df["phone_number"].tolist()
    rand_card_numbers = df["card_number"].tolist()
    rand_order_numbers = df["order_number"].tolist()
    
    return rand_names, rand_addresses, rand_phone_numbers, rand_card_numbers, rand_order_numbers

def json_to_txt(path):
    assert os.path.exists(path)
    fle = os.listdir(path)
    save_dir = f"../data/simdial/{path.split('/')[-1]}"
    assert len(fle) == 1, f'{path} has {len(fle)} jsons, please delete the ones you do not want'
    if len(os.listdir(save_dir)) != 0:
        print(f'{save_dir} is not empty, deleting existing files...')
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))
    
    with open(os.path.join(path, fle[0]), 'r') as fh:
        data = json.load(fh)
    
    for i, dial in enumerate(data['dialogs']):
        lines = []
        for turn in dial:
            lines.append(f"{turn['speaker']}: {turn['utt']}\n")
        with open(f"{save_dir}/dial-{i}.txt", 'w') as fh:
            fh.writelines(lines)


if __name__ == "__main__":
    # pipeline here
    # generate a fix 500 test set and 5000 training set.
    # generate them separately so the model can choose a subset for train and
    # test on all the test set to see generalization.

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain")
    parser.add_argument("--complexity")
    parser.add_argument("--train_size",type=int)
    parser.add_argument("--valid_size",type=int)
    parser.add_argument("--test_size",type=int)
    parser.add_argument("--num_info_ask",type=int,default=1)
    parser.add_argument("--one_token_private_info", default='false')
    parser.add_argument("--save_dir")
    args = parser.parse_args()

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    test_size = args.test_size
    valid_size = args.valid_size
    train_size = args.train_size
    gen_bot = Generator()


    rest_spec = RestSpec()
    rest_style_spec = RestStyleSpec()
    rest_pitt_spec = RestPittSpec()
    bus_spec = BusSpec()
    movie_spec = MovieSpec()
    weather_spec = WeatherSpec()

    if args.num_info_ask and args.num_info_ask in [1,2,3]:
        num_info_ask = args.num_info_ask
    else:
        num_info_ask = 1

    track_package_spec = TrackPackageSpec(args.one_token_private_info, num_info_ask)

    domain_specs = {
        "restaurant": rest_spec,
        "restaurant_style": rest_style_spec,
        "rest_pitt": rest_pitt_spec,
        "bus": bus_spec,
        "weather": weather_spec,
        "movie": movie_spec,
        "track_package": track_package_spec
    }

    complexity_types = {
        "mix": complexity.MixSpec,
        "clean": complexity.CleanSpec
    }
  
  
    gen_bot.gen_corpus(save_dir+"/test", domain_specs[args.domain], complexity_types[args.complexity], test_size)
    json_to_txt(save_dir+"/test")

    gen_bot.gen_corpus(save_dir+"/valid", domain_specs[args.domain], complexity_types[args.complexity], valid_size)
    json_to_txt(save_dir+"/valid")

    gen_bot.gen_corpus(save_dir+"/train", domain_specs[args.domain], complexity_types[args.complexity], train_size)
    json_to_txt(save_dir+"/train")

