# SimDial Privacy Synthetic Dialog Generator

This project is built upon the CMU SimDial project. Changes has been made to create dialogs for the privacy project.


## Background

### CMU SimDial: Synthetic Task-oriented Dialog Generator with Controllable Complexity
This is the dialog data used 
by our SIGDIAL 2018 paper: [Zero-Shot Dialog Generation with Cross-Domain Latent Actions](https://arxiv.org/abs/1805.04803). 
See paper for details. The source code and data used for the paper can be found at [here](https://github.com/snakeztc/NeuralDialog-ZSDG).

## Prerequisites

 - Python 3.8.0+
 - Numpy 1.16.6
 - NLTK
 - progressbar2 3.53.1
 - Faker  6.6.2
 - names  0.3.0
 
 
## Usage 
Run the following code to generate dialog data for a selected domain that is defined in the  *multiple_domains.py* script. 
  
    python multiple_domains.py --domain {domain_name} --complexity {complexity_type} --train_size {train_size} --test_size {test_size} --one_token_private_info {use_one_token_private_info_or_not} --save_dir {output_dir}

For example

    python multiple_domains.py --domain track_package --complexity mix --train_size 500 --test_size 200 --one_token_private_info false --save_dir output

Defined domains in the original paper include:
- `restaurant`  
- `restaurant_style`
- `rest_pitt`
- `bus`
- `weather`
- `movie`

Defined domains for this project include:
- `track_package`
- `return_package` (pending)

Complexity types, as defined in the original paper, include:
- `mix`
- `clean`

The data will be saved into two folders within the assigned out_dir
- save_dir/test/ for testing data 
- save_dir/train/ for training data


