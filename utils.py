import re
import torch

def detect_private_tokens(text, token_ids):
    def is_digit(token):
        return token.isdigit()
    
