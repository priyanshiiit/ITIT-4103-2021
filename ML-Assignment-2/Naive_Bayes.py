import random
from collections import defaultdict

def normalize(probs):
    sum_probs = sum(probs)
    if sum_probs == 0:          # smoothing is done here, if sum_of_probs = 0, means that one or more conditional probabilities are 0 
        return normalize([1 for each in probs])
    return [each/sum_probs for each in probs]

def normalize_dict(d):
    total = sum(d.values())
    for each in d:
        d[each] /= total
    return d

def train(train_data):
    # returns a belief
    prior = defaultdict(int)
    counter = {i: {x: [0, 0] for x in range(10)} for i in range(784)}
    for data in train_data:
        data = convert_data(data)
        label = data[0]
        pixels = data[1:]
        prior[label] += 1
        for i in range(len(pixels)):
            counter[i][label][pixels[i]] += 1
    prior = normalize_dict(prior)
    for i in range(784):
        for x in range(10):
            counter[i][x] = normalize(counter[i][x])
    return prior, counter

def argmax(l):
    maximum = 0
    result = None
    for i in range(len(l)):
        if l[i] > maximum:
            maximum = l[i]
            result = i
    return result

# returns int
def predict(data):
    global belief, prior
    probs = [prior[x] for x in range(10)]
    data = convert_data(data)
    for i in range(len(data)):
        for x in range(10):
            probs[x] *= belief[i][x][data[i]]
        probs = normalize(probs)
    return argmax(probs)

def convert_data(data):
    return [data[0]] + [1 if each > 0 else 0 for each in data[1:]]


def run_test(data):
    
    right_output = 0

    for i in range(len(data)):
        row = data[i]
        label = row[0]
        pixels = row[1:]
        prediction = predict(pixels)

        if label==prediction:
            right_output+=1
    acc = right_output/10000
    return acc

    
    
        

if __name__=="__main__":
    train_data = [list(map(int, line.strip().split(","))) for line in open("mnist_train.csv").read().strip().split("\n")]
    prior, belief = train(train_data)
    
    test_data = [list(map(int, line.strip().split(","))) for line in open("mnist_test.csv").read().strip().split("\n")]
    accuracy = run_test(test_data)
    print("Accuracy: {}".format(accuracy))