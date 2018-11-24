# Concept learner.


# Algorithm 4.2
# Find Least General Generalization f a set of instances
def lgg_conj(x, y):
    z = []
    for i in range(0, len(x)):
        if x[i] != y[i]:
            z.append(0)
        else:
            z.append(x[i])
    return z


# Initialize the learner with an array of instances.

# DATA LOADING FROM SPAMBASE
data = open("spambase.csv", "r")
instances = [line.rstrip('\n') for line in data]

# DATA HANDLING

# For each instance, split it
ref_instances = []
for i in range(0, len(instances)):
    ref_instances.append(instances[i].split(","))
# Now I have ref_instances containing a matrix of raw data.

# I only consider spam data.
spam_ref_instances = [spam_instance for spam_instance in ref_instances if spam_instance[57] == '1']

# Calculate average
avg_length_capital_spam = 0
avg_length_capital_ham = 0
for ref_instance in ref_instances:
    if ref_instance[57] == '1':
        avg_length_capital_spam += float(ref_instance[27])
        if ref_instance[27] != '0':
            print(ref_instance[27])
    else:
        avg_length_capital_ham += float(ref_instance[27])
print("Avg spam:" + str(avg_length_capital_spam / len(spam_ref_instances)))
print("Avg ham:" + str(avg_length_capital_ham / (len(ref_instances) - len(spam_ref_instances))))


# For each spam instance, define an array containing 0 and 1, depending on information inside.
for ref_instance in spam_ref_instances:
    # Percentage of word matching a particular word.
    for i in range(0, 48):
        if float(ref_instance[i]) > 1:
            ref_instance[i] = 1
        else:
            ref_instance[i] = -1
    # Percentage of characters in the e-mail that match CHAR
    for i in range(48, 54):
        if float(ref_instance[i]) > 0.01:
            ref_instance[i] = 1
        else:
            ref_instance[i] = -1
    # Average length of uninterrupted sequences of capital letters (AVG spam=9.519, ham=2.377)
    if float(ref_instance[54]) >= 1:
        ref_instance[54] = 1
    else:
        ref_instance[54] = -1
    # Length of longest uninterrupted sequence of capital letters (AVG spam=104.39, ham=18.21)
    if float(ref_instance[55]) >= 1:
        ref_instance[55] = 1
    else:
        ref_instance[55] = -1
    # Total number of capital letters in the e-mail (AVG spam=470.61, ham=161.47)
    if float(ref_instance[56]) >= 2:
        ref_instance[56] = 1
    else:
        ref_instance[56] = -1

# Algorithm 4.1
x = spam_ref_instances.pop()
H = x
while len(spam_ref_instances) != 0:
    x = spam_ref_instances.pop()
    H = lgg_conj(H, x)
print(H)
