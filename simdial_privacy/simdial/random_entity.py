import names
import random
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
        self.used_values.addresses.add(candidate)
        return candidate
    
    def generate_cred_card(self):
        fake = Faker()
        candidate = fake.credit_card_number()
        while candidate in self.used_values.card_numbers:
            candidate = fake.credit_card_number()
        self.used_values.card_numbers.add(candidate)
        return candidate



us = UsedSlotValues()
generator = GenerateRandomSlotValue(us)
print(generator.generate_address())

    
  
  

