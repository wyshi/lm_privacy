import re
import torch
import pandas as pd
from transformers import GPT2Tokenizer
from transformers import RobertaTokenizer
from spacy import displacy
import en_core_web_sm

nlp = en_core_web_sm.load()

def detect_private_tokens(dialog, domain):
    """
    Detect private information in the original dialog string
    """
    def get_phone(sent):
        phone_num_template = re.compile(r'\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}')
        m = phone_num_template.search(sent)
        if m is not None:
            return m.group()

    def get_order_num(sent):
        order_num_template = re.compile(r'\(?\d{3}\)?[-]?\d{5}[-]?\d{4}')
        m = order_num_template.search(sent)
        if m is not None:
            return m.group()
    
    def get_address(sent):
        if sent.startswith("My address is "):
            return sent[len("My address is "):-1]
        elif sent.startswith("Ok, it is "):
            return sent.split[len("Ok, it is "):-1]
        else:
            return sent[:-1]

    print("Length of original dialog string:", len(dialog))

    IS_ASK_ADDRESS = False
    private_tokens = []
    dialog_by_speaker = dialog.strip().split("\n")
    db = pd.read_csv("database/database_500.csv")
    address_db = db['address'].tolist()

    # recognize dialog order, assume the turns are 1-1, could generalize later
    if dialog.startswith("SYS"):
        orders = [1,2]
    elif dialog.startswith("USR"):
        orders = [2,1]
    else:
        raise NotImplementedError("Dialog not following the correct template")


    for i in range(len(dialog_by_speaker)):
        # If it is user's turn, check private info 
        sent = dialog_by_speaker[i]
        if orders[i%2] == 2:
            docs = nlp(sent)
            entities = [(i.text, i.label_, i.start_char, i.end_char) for i in docs.ents]
            # First check detectable entities
            if IS_ASK_ADDRESS:
                IS_ASK_ADDRESS = False

                for ad in address_db:
                    if ad in sent:
                        #has_address = get_address(sent[4:].strip())
                        print("PRIVATE INFO:", ad)
                        private_tokens.append(ad)

            elif entities != []:
                for ent in entities:
                    if ent[1] in ['ORG','PERSON','LOC']:
                        print("PRIVATE INFO:", ent[0])
                        private_tokens.append(ent[0])

            # Then check non-detectable entities by rules
            has_phone = get_phone(sent)
            if has_phone:
                print("PRIVATE INFO:", has_phone)
                private_tokens.append(has_phone)

            has_order_number = get_order_num(sent)
            if has_order_number:
                print("PRIVATE INFO:", has_order_number)
                private_tokens.append(has_order_number)

        # For address, we check for template for now. will generalize later
        elif "address" in sent:
            IS_ASK_ADDRESS = True


    return private_tokens


def private_token_classifier(dialog, domain, tokenizer):
    """
    Detect private tokens in a dialog with a selected tokenizer

    :param dialog: dialog history in a string
    :param tokenizer: a transformer tokenizer
    :return: a vector representing whether each token is private
    """

    if domain != "track_package":
        raise NotImplementedError("Only support track package domain now")
        
    private_tokens = detect_private_tokens(dialog, domain)
    print("Private Token List", private_tokens)
    

    tokens = tokenizer.tokenize(dialog)
    print("Encoded Tokens", tokens)

    encoded_labels = []
    queue = private_tokens.copy()
    for t in range(len(tokens)):
        lab = 0
        curr_enc_token = tokens[t].strip()
        if queue != []:
            curr_private_token = queue[0].strip()
        
            if curr_enc_token != "" and curr_private_token.startswith(curr_enc_token):
                print("Private Token exists, Case 1: ",curr_private_token,curr_enc_token)
                curr_private_token = curr_private_token[len(curr_enc_token):]
                lab = 1
            elif curr_enc_token[1:] != "" and curr_private_token.startswith(curr_enc_token[1:]):
                print("Private Token exists, Case 2: ",curr_private_token,curr_enc_token[1:])
                curr_private_token = curr_private_token[len(curr_enc_token)-1:]
                lab = 1
            
            if curr_private_token != "":
                queue[0] = curr_private_token
            else:
                queue = queue[1:]
 
        encoded_labels.append(lab)

    print(list(zip(tokens,encoded_labels)))
    return encoded_labels

    
    
    
    


example_input = """SYS: Hello, I am with customer support bot. What can I do for you?
USR: Hi. I placed an order but I don't know if it has been shipped.
SYS: Can you verify your full name so I can look that up?
USR: Michael Park
SYS: Ok, let me get your phone number really quick.
USR: You can reach me at 347-367-3553.
SYS: Verify your order number please.
USR: Sure, it is 358-76768-2727
SYS: We will need the shipping address as well.
USR: My address is 226 Nicole Ramp Apt. 929 Prestonfurt, UT 29527.
SYS: The tracking number of your package is 71. What else can I do?
USR: All good. See you.
SYS: It was a great pleasure helping you."""

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
private_token_classifier(example_input, "track_package", tokenizer)

