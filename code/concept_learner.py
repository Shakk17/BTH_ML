# Concept learner.

import random


# Algorithm 4.2
# Find Least General Generalization f a set of instances
def lgg_conj(x, y):
    z = []
    for i in range(len(x)):
        if x[i] != y[i]:
            z.append(0)
        else:
            z.append(x[i])
    return z


def compare(c_instance):
    # Percentage of word matching a particular word.
    for i in range(48):
        if float(c_instance[i]) > 2:
            c_instance[i] = 1
        else:
            c_instance[i] = -1
    # Percentage of characters in the e-mail that match CHAR
    for i in range(48, 54):
        if float(c_instance[i]) > 0.01:
            c_instance[i] = 1
        else:
            c_instance[i] = -1
    # Average length of uninterrupted sequences of capital letters (AVG spam=9.519, ham=2.377)
    if float(c_instance[54]) >= 1:
        c_instance[54] = 1
    else:
        c_instance[54] = -1
    # Length of longest uninterrupted sequence of capital letters (AVG spam=104.39, ham=18.21)
    if float(c_instance[55]) >= 1:
        c_instance[55] = 1
    else:
        c_instance[55] = -1
    # Total number of capital letters in the e-mail (AVG spam=470.61, ham=161.47)
    if float(c_instance[56]) >= 2:
        c_instance[56] = 1
    else:
        c_instance[56] = -1
    return c_instance


# Initialize the learner with an array of instances.

# DATA LOADING FROM SPAMBASE
data = open("data\spambase.csv", "r")
features = data.readline().split(",")
text_rows = [line.rstrip('\n') for line in data]

# For each instance, split it
instances = []
for i in range(0, len(text_rows)):
    instances.append(text_rows[i].split(","))

# Now I have instances containing a matrix of raw data.
random.shuffle(instances)
# Divide dataset in training set and dataset.
train_test_ratio = 0.80
train_size = int(len(instances) * train_test_ratio)
train_set = instances[:train_size]
test_set = instances[train_size:]

# I only consider spam data.
spam_instances = [spam_instance for spam_instance in train_set if spam_instance[57] == '1']

# For each spam instance, define an array containing 0, -1, 1, depending on information inside.
for row in range(len(spam_instances)):
    spam_instances[row] = compare(spam_instances[row])

# Algorithm 4.1
x = spam_instances[0]
hypothesis = x
for i in range(1, len(spam_instances)):
    x = spam_instances[i]
    hypothesis = lgg_conj(hypothesis, x)

# Print conjunction rule.
i = 0
print("+++ CONJUNCTION RULE +++")
for literal in hypothesis:
    if literal == -1:
        print("{} = Few".format(features[i]))
    elif literal == 1:
        print("{} = A lot".format(features[i]))
    i += 1

# Test results.
t_pos = 0
f_pos = 0
t_neg = 0
f_neg = 0
total_pos = len([instance for instance in test_set if instance[57] == '1'])
total_neg = len([instance for instance in test_set if instance[57] == '0'])
for instance in test_set:
    spam = True
    # Compare instance of test set with hypothesis.
    for i in range(len(compare(instance))-1):
        if hypothesis[i] != 0 and (int(hypothesis[i]) != int(instance[i])):
            spam = False
    if spam:
        if instance[57] == '1':
            t_pos += 1
        else:
            f_pos += 1
    else:
        if instance[57] == '0':
            t_neg += 1
        else:
            f_neg += 1
pos = t_pos + f_pos
neg = t_neg + f_neg
print("\nTrue Positive: {} / {} -> {:.2f}%".format(t_pos, pos, t_pos/pos*100))
print("False Positive: {} / {} -> {:.2f}%".format(f_pos, pos, f_pos/pos*100))
print("True Negative: {} / {} -> {:.2f}%".format(t_neg, neg, t_neg/neg*100))
print("False Negative: {} / {} -> {:.2f}%\n".format(f_neg, neg, f_neg/neg*100))
print("Accuracy: {} / {} -> {:.2f}%".format(
    t_pos+t_neg,
    len(test_set),
    (t_pos+t_neg)/len(test_set)*100))
print("Precision: {} / {} -> {:.2f}%".format(
    t_pos,
    pos,
    t_pos/pos*100
))
