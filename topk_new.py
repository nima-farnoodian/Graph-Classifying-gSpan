"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from sklearn import metrics

from gspan_mining import gSpan
from gspan_mining import GraphDatabase
from operator import itemgetter, attrgetter 

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
        #self.selector=k_selector(k) # keeps the top-k patterns with high confidence 
        self.minsup = minsup
        self.gid_subsets = subsets
        self.current_prune=True
        self.patterns=[]

    # Stores any pattern found that has not been pruned
    def store(self, dfs_code, gid_subsets):
        p=len(gid_subsets[0])
        n=len(gid_subsets[1])
        total=p+n
        if total==0:
            confidence=0
        else:
            confidence=p/total
        self.patterns.append(pattern(dfs_code,confidence,total,gid_subsets))
        #self.selector.append(np.round(confidence,5),(dfs_code,total)) # it holds top-k high confident patterns


    # Prunes any pattern that is not frequent in the positive class
    def prune(self, gid_subsets):
        # computing confidence 
        p=len(gid_subsets[0]) 
        n=len(gid_subsets[1])
        total=p+n
        if total==0:
            confidence=0
        else:
            confidence=p/total
        #print(self.selector.score_list)
        #print( confidence not in self.selector.score_link )
       #print(confidence)
        #not self.current_prune
        return total<self.minsup

    # creates a column for a feature matrix
    def create_fm_col(self, all_gids, subset_gids):
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
    def get_feature_matrices(self):
        matrices = [[] for _ in self.gid_subsets]
        for pattern, gid_subsets in self.patterns:
            for i, gid_subset in enumerate(gid_subsets):
                matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
        return [np.array(matrix).transpose() for matrix in matrices]

def top_k():
    """
    Runs gSpan with the specified positive and negative graphs, finds all frequent subgraphs in the positive class
    with a minimum positive support of minsup and prints them.
    """

    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = int(args[3])  # Third parameter: minimum support
    minsup = int(args[4]) # Third parameter: minimum support

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    subsets = [pos_ids, neg_ids]  # The ids for the positive and negative labelled graphs in the database
    task = FrequentPositiveGraphs(minsup,k, graph_database, subsets)  # Creating task

    gSpan(task).run()  # Running gSpan
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
                print(" ")
                break
        print('{} {} {}'.format(dfs_code, confidence, support))

if __name__ == '__main__':
    top_k()
