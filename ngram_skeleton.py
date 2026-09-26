import math, random
from collections import defaultdict

# Run code -> python3 -> from ngram_skeleton import *

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    # add required number of the pad in front of the text
    # n is pad num
    paddedText = start_pad(n) + text
    # return set of the ngram + text after that, then it is repeated until looking all text
    return [(paddedText[i - n:i], paddedText[i]) for i in range(n, len(paddedText))]

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    #  initialization method __init__(self, n, k) which stores
    #  the order n of the model and initializes any necessary internal variables.
    def __init__(self, n, k):
        self.n = n # context length
        self.k = k # add-k smoothing constant
        self.vocab = set() # All character seen in the training
        # counts[context][char] is that counts char followed context
        self.counts = defaultdict(lambda: defaultdict(int))
        # context_totals[context] is that total num of the context
        self.context_totals = defaultdict(int)
        self._sorted_vocab = None # cashed sorted vocab for random_char 

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for context, char in ngrams(self.n, text):
            self.vocab.add(char)
            self.counts[context][char]   += 1
            self.context_totals[context] += 1
        # vocab may have grown, so clear the cache and let it re-sort next time
        self._sorted_vocab = None

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        vocab_size = len(self.get_vocab())
        total = self.context_totals.get(context, 0)
        # if novel context and no smoothing
        if total == 0 and self.k == 0:
            return 1 / vocab_size
        # avoid to make empty entry, check it if there is context or not
        count = self.counts[context][char] if context in self.counts else 0
        # add k-smoothing: (count + k) / (total + k * V)
        return (count + self.k) / (total + self.k * vocab_size)

    def _vocab_list(self):
        ''' Returns the vocab as a sorted list, cached until the next update '''
        if self._sorted_vocab is None:
            self._sorted_vocab = sorted(self.get_vocab())
        return self._sorted_vocab

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        cumulative = 0.0
        vocab = self._vocab_list()
        for char in vocab:
            # accumulative probability mass
            cumulative += self.prob(context, char)
            # return the first character whose cumulative probability exceeds r
            if r < cumulative:
                return char
        return vocab[-1]        

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        # Always start from the n-padding context 
        context = start_pad(self.n)
        result = []
        for _ in range(length):
            char = self.random_char(context)
            result.append(char)
            if self.n > 0:
                # drop the oldest character, add the new one
                context = (context + char)[-self.n:]
        return ''.join(result)

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        pairs = ngrams(self.n, text)
        if not pairs:
            return float('inf')
        log_sum = 0.0
        for context, char in pairs:
            p = self.prob(context, char)
            #  Perplexity is undefined if the language model assigns any zero probabilities to the test set. Then, return "float('inf')" 
            if p <= 0:
                # perplexity can't allow to keep prob == 0, 
                return float('inf')
            # set the sum of logs to avoid underflow
            log_sum += math.log(p)
        # Perplexity = P(text)^(-1/N) = exp(-(1/N) * sum(log p))
        return math.exp(-log_sum / len(pairs))

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        pass

    def get_vocab(self):
        pass

    def update(self, text):
        pass

    def prob(self, context, char):
        pass

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass