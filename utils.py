import re
import torch

def is_digit(texts_lst):
    """
    texts_lst = ["my", " SS", "N", " is", " 123", "456"]
    return: [0, 0, 0, 0, 1, 1]
    """
    is_private = [int(tok.strip().isdigit()) for tok in texts_lst]
    return is_private
    

def split_is_private(is_private, texts):
    #TODO check if the odd-number-one is always private, and even-number-one is always public
    """
    is_private = [0, 1, 1, 0, 0, 0, 1, 0]
    texts = ['name:', 'Rachel', ' Green', '']
    return:
    [[], [0, 1, 1], [0, 0], [0, 1], [0]]
    https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    """
    splits, splits_tgt, splits_01 = [], [], []
    i, j = 0, 0
    # print(is_private)
    while i < len(is_private):
        if is_private[i] == 1:
            j = i + 1
            while j < len(is_private):
                if is_private[j] == 1:
                    j += 1
                else:
                    # j points to 0 now
                    break
            if i != 0:
                 i -= 1 # the non-private token before the private token also needs to be protected
            splits.append(texts[i:j])
            splits_tgt.append(texts[i+1:j+1])
            splits_01.append(is_private[i:j])
            i = j # i, j points to the same 0 now, or hit the end
        else:
            j = i + 1
            while j < len(is_private):
                if is_private[j] == 0:
                    j += 1
                else:
                    # j points to 1 now
                    break
            if j == len(is_private): 
                splits.append(texts[i:j])
                splits_tgt.append(texts[i+1:j+1])
                splits_01.append(is_private[i:j])
            else: 
                if len(is_private[i:j-1]):          
                    splits.append(texts[i:j-1])
                    splits_tgt.append(texts[i+1:j])
                    splits_01.append(is_private[i:j-1])
            i = j # i, j points to the same 1 now, or hit the end    
    # if the first one is private
    if 1 in splits_01[0]: 
        # make sure the odd number ones are always private, by adding an empty [] to the beginning
        splits = [[]] + splits
        splits_tgt = [[]] + splits_tgt
    # deal with the last one
    if len(splits[-1]) > 1:
        splits[-1] = splits[-1][:-1]
    else:
        assert len(splits_tgt[-1]) == 0, f"{splits_tgt[-1]}"
        splits = splits[:-1]
        splits_tgt = splits_tgt[:-1]   
    return list(zip(splits, splits_tgt))
            

def generate_noise(
    private_engine, max_grad_norm, reference, 
) -> torch.Tensor:
    r"""
    Generates a tensor of Gaussian noise of the same shape as ``reference``.

    The generated tensor has zero mean and standard deviation
    sigma = ``noise_multiplier x max_grad_norm ``

    Args:
        max_grad_norm : The maximum norm of the per-sample gradients.
        reference : The reference, based on which the dimention of the
            noise tensor will be determined

    Returns:
        the generated noise with noise zero and standard
        deviation of ``noise_multiplier x max_grad_norm ``
    """
    if private_engine.noise_multiplier > 0 and max_grad_norm > 0:
        return torch.normal(
            0,
            private_engine.noise_multiplier * max_grad_norm,
            reference.shape,
            device=private_engine.device,
            generator=private_engine.random_number_generator,
        )
    return torch.zeros(reference.grad.shape, device=private_engine.device)
