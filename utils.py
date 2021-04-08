import re
import torch
from transformers import GPT2Tokenizer
from transformers import RobertaTokenizer
from spacy import displacy
import en_core_web_sm

nlp = en_core_web_sm.load()
track_package_templates = {"name": {"inform": ["I am %s.", "%s.", "Sure, %s.", "Yes, %s.", "%s"],
                           "request": ["May I have your full name please?", "Can you verify your full name so I can look that up?"]},

                            "phone": {"inform": ["Phone number is %s", "%s.", "You can reach me at %s."],
                                        "request": ["Ok, let me get your phone number really quick.", "Verify your phone number please."]},

                            "address": {"inform": ["My address is %s.", "%s.", "Ok, it is %s."],
                                            "request": ["We will need the shipping address as well.", "Could you please confirm your shipping address?"]},

                            "shipment": {"inform": ["Your package has been delivered.", "Your package will arrive %s."],
                                        "request": ["When can I receive my package", "When will it be delivered?"],
                                        "yn_question": {'status': ["Is it shipped?"],
                                                    'deliver': ["Is it delivered?"]
                                                    }},
                            "order_number": {"inform": ["Sure, it is %s", "%s", "It's %s."],
                                        "request": ["Could you please also provide your order number?", "Verify your order number please.","Can you provide the order number?"]},

                            "default": {"inform": ["The tracking number of your package is %s."],
                                        "request": ["Where is my package?",
                                                    "Could you please help me track my package?",
                                                    "I placed an order but I don't know if it has been shipped."] + ["I ordered a %s several days ago but I can't track it." % k for k in
                                                        ["lipstick", "mobile phone", "pot", "floor lamp", "chair"]]}
                            }


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
    dialog_by_speaker = dialog.strip().split("\n")

    # recognize dialog order, assume the turns are 1-1, could generalize later
    if dialog.startswith("SYS"):
        orders = [1,2]
    elif dialog.startswith("USR"):
        orders = [2,1]
    else:
        raise NotImplementedError("Dialog not following the correct template")

    private_tokens = []
    loc = 0

    for i in range(len(dialog_by_speaker)):
        # If it is user's turn, check private info 
        sent = dialog_by_speaker[i]
        if orders[i%2] == 2:
            docs = nlp(sent)
            entities = [(i, i.label_, i.start_char, i.end_char) for i in docs.ents]
            # First check detectable entities
            if IS_ASK_ADDRESS:
                IS_ASK_ADDRESS = False
                has_address = get_address(sent[4:].strip())
                print("PRIVATE INFO:", has_address)

            elif entities != []:
                for ent in entities:
                    if ent[1] in ['ORG','PERSON','LOC']:
                        print("PRIVATE INFO:", ent[0])

            # Then check non-detectable entities by rules
            has_phone = get_phone(sent)
            if has_phone:
                print("PRIVATE INFO:", has_phone)
            has_order_number = get_order_num(sent)
            if has_order_number:
                print("PRIVATE INFO:", has_order_number)
                
        # For address, we check for template for now. will generalize later
        elif sent[4:].strip() in track_package_templates["address"]["request"]:
            IS_ASK_ADDRESS = True


        loc += len(dialog_by_speaker[i])


def private_token_classifier(dialog, domain, tokenizer):
    """
    Detect private tokens in a dialog with a selected tokenizer

    :param dialog: dialog history in a string
    :param tokenizer: a transformer tokenizer
    :return: a vector representing whether each token is private
    """

    if domain != "track_package":
        raise NotImplementedError("Only support track package domain now")
        
    private_info_by_char = detect_private_tokens(dialog, domain)

    
    
    # encoded_ids = tokenizer.encode(dialog)
    # print("Encoded IDs:", encoded_ids)
    # tokens = tokenizer.tokenize(dialog)
    # print("Tokens", tokens)
    
    


example_input = """SYS: Hello, I am with customer support bot. What do you need?
USR: What's up? I ordered a chair several days ago but I can't track it.
SYS: May I have your full name please?
USR: I am David Scales.
SYS: Verify your phone number please.
USR: 1606220154.
SYS: Verify your order number please.
USR: 849-37907-2687
SYS: Could you please confirm your shipping address?
USR: My address is 6448 Paula Turnpike North Ashley, VT 32004.
SYS: The tracking number of your package is 78. Anything else?
USR: That's it. Thanks!"""

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
private_token_classifier(example_input, "track_package", tokenizer)

