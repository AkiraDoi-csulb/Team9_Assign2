import math, random, os
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
        self.n = n
        self.k = k
        self.models = [NgramModel(i, k) for i in range(n + 1)]
        self.lambdas = [1 / (n + 1)] * (n + 1) 
    def set_lambdas(self, lambdas):
        if len(lambdas) != self.n + 1:
            raise ValueError(f"Expected {self.n + 1}")
        self.lambdas = lambdas

    def get_vocab(self):
        vocab = set()
        for model in self.models:
            vocab.update(model.get_vocab())
        return vocab

    def update(self, text):
        for model in self.models:
            model.update(text)

    def prob(self, context, char):
        interpolated_prob = 0.0
        for i, model in enumerate(self.models):
            sub_context = context[-i:] if i > 0 else ''
            interpolated_prob += self.lambdas[i] * model.prob(sub_context, char)
        return interpolated_prob

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    print("Ngram Model with n=1, k=0")
    m1 = NgramModelWithInterpolation(1, 0)
    m1.update('abab')
    print(m1.prob('a', 'a')) 
    print(m1.prob('a', 'b')) 

    print("Ngram Model with n=2, k=1")
    m2 = NgramModelWithInterpolation(2, 1)
    m2.update('abab')
    m2.update('abcd')
    print(m2.prob('~a', 'b')) 
    print(m2.prob('ba', 'b')) 
    print(m2.prob('~c', 'd')) 
    print(m2.prob('bc', 'd')) 
    print("Setting lambdas to [0.1, 0.2, 0.7] for interpolation...")
    m2.set_lambdas([0.1, 0.2, 0.7])
    print(m2.prob('~a', 'b')) 
    print(m2.prob('ba', 'b')) 
    print(m2.prob('~c', 'd')) 
    print(m2.prob('bc', 'd'))

    print("Ngram Model with n=3, k=0.5")
    m3 = NgramModelWithInterpolation(3, 0.5) 
    m3.update('abab')
    print(m3.prob('~a', 'b'))
    print(m3.prob('ba', 'b'))
    print(m3.prob('~c', 'd'))
    print(m3.prob('bc', 'd'))
    print("Setting k back to 1 for smoothing...")
    m3.k = 1
    print(m3.prob('~a', 'b'))
    print(m3.prob('ba', 'b'))
    print(m3.prob('~c', 'd'))
    print(m3.prob('bc', 'd'))
    print("Setting lambdas to [0.05, 0.15, 0.30, 0.50] for interpolation...") 
    m3.set_lambdas([0.05, 0.15, 0.30, 0.50])
    print(m3.prob('~a', 'b'))
    print(m3.prob('ba', 'b'))
    print(m3.prob('~c', 'd'))
    print(m3.prob('bc', 'd'))


    def update_model_from_file_lines(model, path):
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                cleaned = line.strip()
                if cleaned:
                    model.update(cleaned)
        return model

    def create_cities_ngrams(folderpath,model_class, n, k):
        model = model_class(n, k)
        for filename in os.listdir(folderpath):
            if filename in COUNTRY_CODES:
                update_model_from_file_lines(model, os.path.join(folderpath, filename))

        return model

    def create_country_model(file_path, model_class, n, k, lambdas=None):
        model = model_class(n, k)
        if lambdas and hasattr(model, 'set_lambdas'):
            model.set_lambdas(lambdas)
        update_model_from_file_lines(model, file_path)
        return model

    def predict_country(city_name, country_models):
        best_country = None
        best_log_likelihood = float('-inf')

        for code, model in country_models.items():
            log_likelihood = 0.0
            pairs = ngrams(model.n, city_name)

            for context, char in pairs:
                p = model.prob(context, char)
                if p > 0:
                    log_likelihood += math.log(p)
                else:
                    log_likelihood += float('-inf')

            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_country = code

        return best_country if best_country else COUNTRY_CODES[0]

    def evaluate_validation(val_path, country_models):
        total_samples = 0
        correct_predictions = 0

        for code in COUNTRY_CODES:
            val_file = os.path.join(val_path, code)
            if not os.path.exists(val_file):
                val_file += '.txt'

            if os.path.exists(val_file):
                with open(val_file, encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        city = line.strip()
                        if city:
                            pred = predict_country(city, country_models)
                            if pred == code:
                                correct_predictions += 1
                            total_samples += 1

        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0.0
        print(f"Validation Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")
        return accuracy

    n = 3
    k = 1
    lambdas = [0.05, 0.15, 0.30, 0.50] 

    train_path = "cities_train/train" 
    val_path   = "cities_val/val"
    output_file     = "test_labels.txt"
    test_file       = "cities_test.txt"

    country_models = {}

    for code in COUNTRY_CODES:
        file_path = os.path.join(train_path, code)
        if not os.path.exists(file_path):
            file_path += '.txt'

        if os.path.exists(file_path):
            model = create_country_model(
                file_path,
                NgramModelWithInterpolation,
                n=n,
                k=k,
                lambdas = lambdas
            )
            country_models[code] = model
            print(f"Loaded and trained model for country: {code.upper()}")
        else:
            print(f"Warning: File for country '{code}' not found at path: {file_path}")

    global_vocab = set()
    for m in country_models.values():
        global_vocab.update(m.get_vocab())
    for m in country_models.values():
        for sub_m in getattr(m, 'models', [m]):
            sub_m.vocab = global_vocab

    # 3. Evaluate Validation Set (OUTSIDE the country training loop)
    if os.path.exists(val_path) or os.path.exists(val_path + ".txt"):
        print("\n--- Evaluating Model Performance on Validation Set ---")
        evaluate_validation(val_path, country_models)

    if os.path.exists(test_file):
            print(f"Generating '{output_file}'...")
            with open(test_file, encoding='utf-8', errors='ignore') as infile, \
                open(output_file, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    city = line.strip()
                    if city:
                        pred = predict_country(city, country_models)
                        outfile.write(f"{pred}\n")
            print(f"Saved '{output_file}' in project root.")
    else:
            print(f"Error: Target file '{test_file}' not found in current directory.")
   