"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from sklearn import naive_bayes
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from gspan_mining import gSpan
from gspan_mining import GraphDatabase
from operator import itemgetter, attrgetter 

from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class pattern:
    def __init__(self, code, confidence, support, gid_subsets):
        self.code = code
        self.confidence = confidence
        self.support=support
        self.gid_subsets = gid_subsets
    def __repr__(self):
        return repr((self.code, self.confidence, self.support,self.gid_subsets))

class k_selector:
    """Utility class to control k-top elements."""
    def __init__(self, k):
        self.score_list = list() # a list of top k (or less than k) scores that have been observed so far
        self.score_link = dict() # a dictionary where keys are the scores and the values are the sequences whose scores are equale to the key
        self.k=k # the number of unique scores
    def append(self,score,sequence):
    
        '''
        Input: 
            score e.g, total support or Wracc
            obtained sequence
        Output:
            return True if the sequence could be added to the top-k list sequences
        '''
        ret=False
        if score not in self.score_link:
            if len(self.score_list)<self.k:
                self.score_list.append(score)
                self.score_link[score]=[]
                self.score_link[score].append(sequence)
                ret=True
            else:
                minimum=np.min(self.score_list)
                if score>minimum:
                    self.score_list.pop(self.score_list.index(minimum))
                    del(self.score_link[minimum])
                    self.score_link[score]=[]
                    self.score_list.append(score)
                    self.score_link[score].append(sequence)
                    ret=True
                if score==minimum:
                    self.score_link[score].append(sequence)
                    ret=True
        else:
            self.score_link[score].append(sequence)
            ret=True
        return ret



class PatternGraphs:
    """
    This template class is used to define a task for the gSpan implementation.
    You should not modify this class but extend it to define new tasks
    """

    def __init__(self, database):
        # A list of subsets of graph identifiers.
        # Is used to specify different groups of graphs (classes and training/test sets).
        # The gid-subsets parameter in the pruning and store function will contain for each subset, all the occurrences
        # in which the examined pattern is present.
        self.gid_subsets = []

        self.database = database  # A graphdatabase instance: contains the data for the problem.

    def store(self, dfs_code, gid_subsets):
        """
        Code to be executed to store the pattern, if desired.
        The function will only be called for patterns that have not been pruned.
        In correlated pattern mining, we may prune based on confidence, but then check further conditions before storing.
        :param dfs_code: the dfs code of the pattern (as a string).
        :param gid_subsets: the cover (set of graph ids in which the pattern is present) for each subset in self.gid_subsets
        """
        print("Please implement the store function in a subclass for a specific mining task!")

    def prune(self, gid_subsets):
        """
        prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
        should be pruned.
        :param gid_subsets: A list of the cover of the pattern for each subset.
        :return: true if the pattern should be pruned, false otherwise.
        """
        print("Please implement the prune function in a subclass for a specific mining task!")


class FrequentPositiveGraphs(PatternGraphs):
    """
    Finds the frequent (support >= minsup) subgraphs among the positive graphs.
    This class provides a method to build a feature matrix for each subset.
    """

    def __init__(self, minsup, k, database, subsets):
        """
        Initialize the task.
        :param minsup: the minimum positive support
        :k: the size of topk 
        :param database: the graph database
        :param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
        """
        super().__init__(database)
        #self.patterns = []  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
        self.minsup = minsup
        self.gid_subsets = subsets
        self.patterns=[]
        self.total_supp=len(subsets[0])+len(subsets[2])
        print(self.total_supp)

    # Stores any pattern found that has not been pruned
    def store(self, dfs_code, gid_subsets):
        p=len(gid_subsets[0])
        n=len(gid_subsets[2])
        total=p+n
        if total==0:
            confidence=0
        else:
            confidence=p/total
        self.patterns.append(pattern(dfs_code,confidence,total,gid_subsets))

    # Prunes any pattern that is not frequent in the positive class
    def prune(self, gid_subsets):
        # computing confidence 
        p=len(gid_subsets[0]) 
        n=len(gid_subsets[2])
        total=p+n
        if total==0:
            confidence=0
        else:
            confidence=p/total
        #print(self.selector.score_list)
        #print(confidence not in self.selector.score_link )
       #print(confidence)
        #not self.current_prune
        return total/self.total_supp<self.minsup
    
def get_output(task,k):
    selected_patterns=[]
    sort=sorted(task.patterns, key=attrgetter('confidence', 'support'),reverse=True)
    bestConf = -1
    bestSupp = -1
    for patt in sort:
        confidence=patt.confidence
        support=patt.support
        dfs_code=patt.code
        if (confidence != bestConf or support != bestSupp):
            bestConf = confidence
            bestSupp = support
            k-=1
            if k==-1:
                #print(" ")
                break
        #print('{} {} {}'.format(dfs_code, confidence, support))
        selected_patterns.append((dfs_code,patt.gid_subsets))
        #selected_patterns.append((dfs_code,patt.gid_subsets,confidence,support))
    #selected_patterns = sorted( selected_patterns, key=lambda x:(x[2], x[3]), reverse = True)
    #selected_patterns=[(patt,gid) for patt,gid,_,_ in selected_patterns]
    return selected_patterns

# creates a column for a feature matrix
def create_fm_col(all_gids, subset_gids):
    subset_gids = set(subset_gids)
    bools = []
    for i, val in enumerate(all_gids):
        if val in subset_gids:
            bools.append(1)
        else:
            bools.append(0)
    return bools

# return a feature matrix for each subset of examples, in which the columns correspond to patterns
# and the rows to examples in the subset.
def get_feature_matrices(task,patterns):
    matrices = [[] for _ in task.gid_subsets]
    for pattern, gid_subsets in patterns:
        for i, gid_subset in enumerate(gid_subsets):
            matrices[i].append(create_fm_col(task.gid_subsets[i], gid_subset))
    return [np.array(matrix).transpose() for matrix in matrices]

def train_and_evaluate(minsup,k, database, subsets):
    task = FrequentPositiveGraphs(minsup,k, database, subsets)  # Creating task

    gSpan(task).run()  # Running gSpan
    patterns=get_output(task,k)
    # Creating feature matrices for training and testing:
    
    features = get_feature_matrices(task,patterns)
    train_fm = np.concatenate((features[0], features[2]))  # Training feature matrix
    train_labels = np.concatenate((np.full(len(features[0]), 1, dtype=int), np.full(len(features[2]), -1, dtype=int)))  # Training labels
    test_fm = np.concatenate((features[1], features[3]))  # Testing feature matrix
    test_labels = np.concatenate((np.full(len(features[1]), 1, dtype=int), np.full(len(features[3]), -1, dtype=int)))  # Testing labels

    #classifier =DecisionTreeClassifier(random_state=1)  # Creating model object
    mlp = MLPClassifier(random_state=1, max_iter=500) # Creating model object
    svc=SVC(probability=True)
    rf=RandomForestClassifier(max_depth=2, random_state=0)
    classifier = VotingClassifier(estimators=[('mlp', mlp), ('svc', svc),('rf',rf)], voting='soft')
    classifier.fit(train_fm, train_labels)  # Training model

    predicted = classifier.predict(test_fm)  # Using model to predict labels of testing data

    accuracy = metrics.accuracy_score(test_labels, predicted)  # Computing accuracy:

    # Printing frequent patterns along with their positive support:
    #print("number of patterns:", len(patterns))
    for pattern, gid_subsets in patterns:
        p = len(gid_subsets[0])
        n=len(gid_subsets[2])
        total=p+n
        if total==0:
            confidence=0
        else:
            confidence=p/total
        print('{} {} {}'.format(pattern, confidence,total))
    # printing classification results:
    print(predicted.tolist())
    print('accuracy: {}'.format(accuracy))
    print()  # Blank line to indicate end of fold.
    
def example2():
    """
    Runs gSpan with the specified positive and negative graphs; finds all frequent subgraphs in the training subset of
    the positive class with a minimum support of minsup.
    Uses the patterns found to train a naive bayesian classifier using Scikit-learn and evaluates its performances on
    the test set.
    Performs a k-fold cross-validation.
    """

    args = sys.argv
    database_file_name_pos = args[1] # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = 15 # top_k
    minsup = 0.2# Third parameter: minimum support
    nfolds=int(args[3])

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos)   )
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    # If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
    if nfolds < 2:
        subsets = [
            pos_ids,  # Positive training set
            pos_ids,  # Positive test set
            neg_ids,  # Negative training set
            neg_ids  # Negative test set
        ]
        # Printing fold number:
        print('fold {}'.format(1))
        train_and_evaluate(minsup, k,graph_database, subsets)

    # Otherwise: performs k-fold cross-validation:
    else:
        pos_fold_size = len(pos_ids) // nfolds
        neg_fold_size = len(neg_ids) // nfolds
        for i in range(nfolds):
            # Use fold as test set, the others as training set for each class;
            # identify all the subsets to be maintained by the graph mining algorithm.
            subsets = [
                np.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),  # Positive training set
                pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
                np.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),  # Negative training set
                neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
            ]
            # Printing fold number:
            print('fold {}'.format(i+1))
            train_and_evaluate(minsup,k, graph_database, subsets)


if __name__ == '__main__':
	#example1()
	example2()

