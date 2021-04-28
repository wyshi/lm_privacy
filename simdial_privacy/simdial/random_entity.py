import names
import random
import os
import pandas as pd
from faker import Faker

class UsedSlotValues(object):
    def __init__(self):
        self.names = set()
        self.addresses = set()
        self.phone_numbers = set()
        self.card_numbers = set()

class GenerateRandomSlotValue(object):
    """
    Takes a list of used slot values and generate a new item of selected category without duplication
    """
    def __init__(self, used_values):
        self.used_values = used_values
    
    def generate_name(self):
        candidate = names.get_full_name()
        while candidate in self.used_values.names:
            candidate = names.get_full_name()
        self.used_values.names.add(candidate)
        return candidate
    
    def __generate_phone_helper(self):
        first = str(random.randint(100,999))
        second = str(random.randint(1,888)).zfill(3)
        last = (str(random.randint(1,9998)).zfill(4))

        while last in ['1111','2222','3333','4444','5555','6666','7777','8888']:
            last = (str(random.randint(1,9998)).zfill(4))
        
        # generate the phone number with a random style
        format_style = random.choice(["","-","."])

        return '{}{}{}{}{}'.format(first,format_style,second,format_style,last)
    
    def generate_phone(self):
        candidate = self.__generate_phone_helper()
        while candidate in self.used_values.phone_numbers:
            candidate = self.__generate_phone_helper()
        self.used_values.phone_numbers.add(candidate)
        return candidate
    
    def generate_address(self):
        fake = Faker()
        candidate = fake.address()
        while candidate in self.used_values.addresses:
            candidate = fake.address()
        candidate = candidate.replace("\n"," ")
        self.used_values.addresses.add(candidate)
        return candidate
    
    def generate_cred_card(self):
        fake = Faker()
        candidate = fake.credit_card_number()
        while candidate in self.used_values.card_numbers:
            candidate = fake.credit_card_number()
        self.used_values.card_numbers.add(candidate)
        return candidate
    
    def generate_order_num(self):
        first = str(random.randint(100,999))
        second = str(random.randint(10000,99999)).zfill(5)
        last = (str(random.randint(1,9998)).zfill(4))
        
        return '{}-{}-{}'.format(first,second,last)


def convert_to_one_token(t):
    return f"<{t.replace(' ','_')}>"
    
def generate_n_rand_entities(n,one_token=False):

    us = UsedSlotValues()
    generator = GenerateRandomSlotValue(us)

    names = []
    addresses = []
    phone_numbers = []
    card_numbers = []
    order_numbers = []

    for i in range(n):
        rand_name = generator.generate_name()
        rand_address = generator.generate_address()
        rand_phone = generator.generate_phone()
        rand_card = generator.generate_cred_card()
        rand_ord = generator.generate_order_num()

        if one_token == True:
            rand_name = convert_to_one_token(rand_name)
            rand_address = convert_to_one_token(rand_address)
            rand_phone = convert_to_one_token(rand_phone)
            rand_card = convert_to_one_token(rand_card)
            rand_ord = convert_to_one_token(rand_ord)

        names.append(rand_name)
        addresses.append(rand_address)
        phone_numbers.append(rand_phone)
        card_numbers.append(rand_card)
        order_numbers.append(rand_ord)
    
    return names, addresses, phone_numbers, card_numbers, order_numbers



if __name__ == "__main__":
    n = 20_000
    names, addresses, phone_numbers, card_numbers, order_numbers = generate_n_rand_entities(n)
    df = pd.DataFrame(list(zip(names, addresses, phone_numbers, card_numbers, order_numbers)), 
                  columns =['name', 'address', 'phone_number', 'card_number', 'order_number'])
    path = os.getcwd()
    save_dir = "database"
    if not os.path.exists(f"{path}/simdial_privacy/{save_dir}"):
        os.mkdir(f"{path}/simdial_privacy/{save_dir}")
    df.to_csv(f"{path}/simdial_privacy/{save_dir}/database_{n}.csv")
    


