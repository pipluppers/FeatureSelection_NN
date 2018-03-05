import pandas as pd
import random

#   Open text files
'''
with open('CS205_SMALLtestdata__68.txt') as file:
    data = file.read()
'''

#   \s is spaces. \s+ is at least one space
df = pd.read_csv('CS205_SMALLtestdata__68.txt', sep = '\s+', names=['Class Label',
        'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6',
        'Feature7', 'Feature8', 'Feature9', 'Feature10'])
#   The head method displays the first five rows
print(df.head())
print('\n', end='')
print(df.columns)
#   Shows some statistics about the data frame, such as count
#print(df.describe())

num_rows = df.shape[0]
num_cols = df.shape[1]

def leave_one_out_cross_validation(dataset):
    accuracy = 20

    return accuracy

def nearest_neighbor(data):
    print("To do")

def feature_search(data):

    #   Dictionaries are easier to search for membership
    current_features = {}

    #   Iterating through the rows
    for i in range(0, 10):
        print('On level %d of search tree' % (i+1))
        best_so_far_accuracy = 0

        #   Iterating through the columns
        for j in range(1, data.shape[1]):
            if not j in current_features:
                print('--Considering adding feature %d' % j)

                accuracy = random.randint(1, 10)
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = j
        #current_features.append(feature_to_add)
        current_features[feature_to_add] = ''
        print('Added feature %d to the current set' % feature_to_add)
    print('\n', current_features, '\n')

feature_search(df)
