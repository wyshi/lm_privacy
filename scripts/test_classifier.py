"""
script to test the classifier
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils 


from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


s = """SYS: Hello, I am with customer support bot. How can I help?
USR: Hello robot. Could you please help me track my package?
SYS: Please provide your full name
USR: Thomas Heller
SYS: Could you please confirm your shipping address?
USR: Yea sure, 36706 Hansen Loop Suite 958 New Marissafort, NC 27982.
SYS: You can track your package using your tracking number, which is LMCICFAGWI. Anything else?
USR: That's it. Thank you.
SYS: Bye.
"""

utils.private_token_classifier(s, "track_package", tokenizer)
