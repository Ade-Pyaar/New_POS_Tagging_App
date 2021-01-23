import string, json, math
import numpy as np


# Punctuation characters
punct = set(string.punctuation)

# Morphology rules used to assign unknown word tokens
A = np.load("A.npy")

B = np.load("B.npy")

with open("tag_counts.json") as fp:

    tag_counts = json.load(fp)
    
with open("vocab.json", "r") as file1: 
        vocab = json.load(file1)

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

    num_tags = len(tag_counts)

    best_probs = np.zeros((num_tags, len(corpus)))

    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)


    s_idx = states.index("--s--")


    for i in range(len(states)):

        if A[s_idx, i] == 0:
            best_probs[i,0] = float('-inf')

        else:
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
    
    num_tags = best_probs.shape[0]
    for i in range(1, len(test_corpus)): 
        if i % 5000 == 0:
            print("Words processed: {:>8}".format(i))
        for j in range(num_tags):
            best_prob_i = float('-inf')
            best_path_i = None
            for k in range(num_tags):
                prob = best_probs[k, i-1] + math.log(A[k, j]) + math.log(B[j, vocab[test_corpus[i]]]) 
                if prob > best_prob_i:
                    best_prob_i = prob
                    best_path_i = k
            best_probs[j,i] = best_prob_i
            best_paths[j,i] = best_path_i
    return best_probs, best_paths



def viterbi_backward(corpus):
    '''
    This function returns the best path.
    
    '''
    
    best_probs, best_paths = initialize(states, tag_counts, A, B, corpus, vocab)
    best_probs, best_paths = viterbi_forward(A, B, corpus, best_probs, best_paths, vocab)
    m = best_paths.shape[1]
    
    z = [None] * m

    num_tags = best_probs.shape[0]
    
    best_prob_for_last_word = float('-inf')
    
    pred = [None] * m
    for k in range(num_tags): 
        if best_probs[k, m-1] > best_prob_for_last_word: 
            best_prob_for_last_word = best_probs[k, m-1]
            z[m - 1] = k
    pred[m - 1] = states[z[m - 1]]
    for i in range(len(corpus)-1, -1, -1):
        pos_tag_for_word_i = z[i]
        z[i - 1] = best_paths[pos_tag_for_word_i, i]
        pred[i - 1] = states[z[i -1]]
    
    return pred
