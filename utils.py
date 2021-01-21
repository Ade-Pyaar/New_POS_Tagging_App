import string, json, math
import numpy as np


# Punctuation characters
punct = set(string.punctuation)

# Morphology rules used to assign unknown word tokens
A = np.load("A.npy")

B = np.load("B.npy")

with open("tag_counts.json") as fp:

    tag_counts = json.load(fp)

noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]
states = ['#', '$', "''", '(', ')', ',', '--s--', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ',

              'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB',

              'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD','VBG', 'VBN', 'VBP', 'VBZ', 'WDT',

              'WP', 'WP$', 'WRB', '``']


def assign_unk(tok):
    """
    Assign unknown word tokens
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"



def get_emission_and_vocab():
    #get vocab
    with open("vocab.json", "r") as file1: 
        vocab = json.load(file1)
        
    #get emission_counts
    with open("emission_counts.json", "r") as file2: 
        emission = json.load(file2)
        
    emission_counts = {}
    
    for key in emission.keys():
        key_l = key.split(' ')
        new_key = (key_l[0], key_l[1])
        emission_counts[new_key] = emission[key]
        
    return vocab, emission_counts



def my_preprocess(sentence):
    """
    Preprocess data
    """
    punct = set(string.punctuation)
    
    orig = []
    prep = []

    # Read data
    file = sentence.split()
    new_file = []
    for word in file:
        for pun in punct:
            word = word.replace(pun,'')
        new_file.append(word)

    for _, word in enumerate(new_file):

        # End of sentence
        if not word.split():
            orig.append(word.strip())
            word = "--n--"
            prep.append(word)
            continue

        # Handle unknown words
        elif word.strip() not in vocab:
            orig.append(word.strip())
            word = assign_unk(word)
            prep.append(word)
            continue

        else:
            orig.append(word.strip())
            prep.append(word.strip())

    return new_file, prep




def predict_pos(prep, emission_counts, vocab):
    '''
    Input: 
        prep: a preprocessed sentence to predict POS for
        emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output: 
        tags: a list of POS tags for prep
    '''

    
    # Get the (tag, word) tuples, stored as a set
    # all_words = set(emission_counts.keys())
    pos_list = []
    
    for word in prep:
        count_final = 0
        pos_final = ''
        
        if word in vocab:
            for pos in states:
                key = (pos, word)
                
                if key in emission_counts.keys():
                    count = emission_counts[key]
                    
                    if count > count_final:
                        count_final = count
                        pos_final = pos
            pos_list.append(pos_final)
            
    return pos_list




def initialize(states, tag_counts, A, B, corpus, vocab):

    '''

    Input: 

        states: a list of all possible parts-of-speech

        tag_counts: a dictionary mapping each tag to its respective count

        A: Transition Matrix of dimension (num_tags, num_tags)

        B: Emission Matrix of dimension (num_tags, len(vocab))

        corpus: a sequence of words whose POS is to be identified in a list 

        vocab: a dictionary where keys are words in vocabulary and value is an index

    Output:

        best_probs: matrix of dimension (num_tags, len(corpus)) of floats

        best_paths: matrix of dimension (num_tags, len(corpus)) of integers

    '''

    # Get the total number of unique POS tags

    num_tags = len(tag_counts)

    

    # Initialize best_probs matrix 

    # POS tags in the rows, number of words in the corpus as the columns

    best_probs = np.zeros((num_tags, len(corpus)))

    

    # Initialize best_paths matrix

    # POS tags in the rows, number of words in the corpus as columns

    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)

    

    # Define the start token

    s_idx = states.index("--s--")

    ### START CODE HERE (Replace instances of 'None' with your code) ###

    

    # Go through each of the POS tags

    for i in range(len(states)): # complete this line

        

        # Handle the special case when the transition from start token to POS tag i is zero

        if A[s_idx, i] == 0: # complete this line

            

            # Initialize best_probs at POS tag 'i', column 0, to negative infinity

            best_probs[i,0] = float('-inf')

        

        # For all other cases when transition from start token to POS tag i is non-zero:

        else:

            

            # Initialize best_probs at POS tag 'i', column 0

            # Check the formula in the instructions above

            best_probs[i,0] = math.log(A[s_idx, i]) + math.log(B[i, vocab[corpus[0]]])

                        

    

    return best_probs, best_paths




def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab):
    '''
    Input: 
        A, B: The transition and emission matrices respectively
        test_corpus: a list containing a preprocessed corpus
        best_probs: an initilized matrix of dimension (num_tags, len(corpus))
        best_paths: an initilized matrix of dimension (num_tags, len(corpus))
        vocab: a dictionary where keys are words in vocabulary and value is an index 
    Output: 
        best_probs: a completed matrix of dimension (num_tags, len(corpus))
        best_paths: a completed matrix of dimension (num_tags, len(corpus))
    '''
    # Get the number of unique POS tags (which is the num of rows in best_probs)
    num_tags = best_probs.shape[0]
    
    # Go through every word in the corpus starting from word 1
    # Recall that word 0 was initialized in `initialize()`
    for i in range(1, len(test_corpus)): 
        
        # Print number of words processed, every 5000 words
        if i % 5000 == 0:
            print("Words processed: {:>8}".format(i))
            
        ### START CODE HERE (Replace instances of 'None' with your code EXCEPT the first 'best_path_i = None') ###
        # For each unique POS tag that the current word can be
        for j in range(num_tags): # complete this line
            
            # Initialize best_prob for word i to negative infinity
            best_prob_i = float('-inf')
            
            # Initialize best_path for current word i to None
            best_path_i = None

            # For each POS tag that the previous word can be:
            for k in range(num_tags): # complete this line
            
                # Calculate the probability = 
                # best probs of POS tag k, previous word i-1 + 
                # log(prob of transition from POS k to POS j) + 
                # log(prob that emission of POS j is word i)
                prob = best_probs[k, i-1] + math.log(A[k, j]) + math.log(B[j, vocab[test_corpus[i]]]) 

                # check if this path's probability is greater than
                # the best probability up to and before this point
                if prob > best_prob_i: # complete this line
                    
                    # Keep track of the best probability
                    best_prob_i = prob
                    
                    # keep track of the POS tag of the previous word
                    # that is part of the best path.  
                    # Save the index (integer) associated with 
                    # that previous word's POS tag
                    best_path_i = k

            # Save the best probability for the 
            # given current word's POS tag
            # and the position of the current word inside the corpus
            best_probs[j,i] = best_prob_i
            
            # Save the unique integer ID of the previous POS tag
            # into best_paths matrix, for the POS tag of the current word
            # and the position of the current word inside the corpus.
            best_paths[j,i] = best_path_i

        ### END CODE HERE ###
    return best_probs, best_paths



def viterbi_backward(corpus):
    '''
    This function returns the best path.
    
    '''
    
    best_probs, best_paths = initialize(states, tag_counts, A, B, corpus, vocab)
    best_probs, best_paths = viterbi_forward(A, B, corpus, best_probs, best_paths, vocab)

    
    # Get the number of words in the corpus
    # which is also the number of columns in best_probs, best_paths
    m = best_paths.shape[1] 
    
    # Initialize array z, same length as the corpus
    z = [None] * m
    
    # Get the number of unique POS tags
    num_tags = best_probs.shape[0]
    
    # Initialize the best probability for the last word
    best_prob_for_last_word = float('-inf')
    
    # Initialize pred array, same length as corpus
    pred = [None] * m
    
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    ## Step 1 ##
    
    # Go through each POS tag for the last word (last column of best_probs)
    # in order to find the row (POS tag integer ID) 
    # with highest probability for the last word
    for k in range(num_tags): # complete this line

        # If the probability of POS tag at row k 
        # is better than the previously best probability for the last word:
        if best_probs[k, m-1] > best_prob_for_last_word: # complete this line
            
            # Store the new best probability for the last word
            best_prob_for_last_word = best_probs[k, m-1]
    
            # Store the unique integer ID of the POS tag
            # which is also the row number in best_probs
            z[m - 1] = k
            
    # Convert the last word's predicted POS tag
    # from its unique integer ID into the string representation
    # using the 'states' list
    # store this in the 'pred' array for the last word
    pred[m - 1] = states[z[m - 1]]
    
    ## Step 2 ##
    # Find the best POS tags by walking backward through the best_paths
    # From the last word in the corpus to the 0th word in the corpus
    for i in range(len(corpus)-1, -1, -1): # complete this line
        
        # Retrieve the unique integer ID of
        # the POS tag for the word at position 'i' in the corpus
        pos_tag_for_word_i = z[i]
        
        # In best_paths, go to the row representing the POS tag of word i
        # and the column representing the word's position in the corpus
        # to retrieve the predicted POS for the word at position i-1 in the corpus
        z[i - 1] = best_paths[pos_tag_for_word_i, i]
        
        # Get the previous word's POS tag in string form
        # Use the 'states' list, 
        # where the key is the unique integer ID of the POS tag,
        # and the value is the string representation of that POS tag
        pred[i - 1] = states[z[i -1]]
        
     ### END CODE HERE ###
    return pred
