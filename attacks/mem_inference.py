import math
import random

def mem_analyze(dataset, increment=1, yes_print=True):
  '''
    DATASET -
        [[text_sample_1, perplexity_1, label_1], 
        [text_sample_2, perplexity_2, label_2], 
        ....                              
        text_sample_N, perplexity_N, label_N]]
    
    increment - 
        how much to increment the cut-off perplexities
      
    yes_print
        whether or not to print each cutoff score and their accuracy rates
    
    output - best cutoff score
  '''
  
  dataset.sort(key = lambda sample: sample[1]) 

  accuracy = {}
  
  accuracy[dataset[1][1]] = [dataset[1][2], 1]
  cutoff = dataset[1][1]

  for sample in dataset[1:]:
  
    prev_cutoff = cutoff
    
    if sample[1] > cutoff:
      cutoff += increment

      accuracy[cutoff] = [0, 0]

      accuracy[cutoff][0] = accuracy[prev_cutoff][0]
      accuracy[cutoff][1] = accuracy[prev_cutoff][1]

      prev_score = accuracy[prev_cutoff][0]/accuracy[prev_cutoff][1]
      accuracy[prev_cutoff].append(prev_score)

    accuracy[cutoff][1] += 1

    accuracy[cutoff][0] += sample[2] #adds one if it is in training data, zero if not

  accuracy.pop(dataset[1][1], None)
  
  max = 0
  max_key = 0
  for cutoff in accuracy:
    score = accuracy[cutoff][0]/accuracy[cutoff][1]
    if score > max:
      max = score
      max_key = cutoff
    if yes_print:
      output = "Under perplexity of | {} | had | {}% | samples memorized\n".format(cutoff, score)
      print(output)
  

  return max_key


fake_dataset = []
for i in range(10000):
  fake_dataset.append([str(i), round(random.uniform(1, 10), 2),  random.randint(0,1)])

best_cutoff = mem_analyze(fake_dataset, increment=0.1, yes_print=true)
print(best_cutoff)
