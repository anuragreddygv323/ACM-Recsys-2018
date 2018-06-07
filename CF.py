# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 19:59:25 2018

RecSys Challenge 2018

@author: lihepeng
"""

import os
import sys
import json
import time

import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, issparse
from sklearn.model_selection import train_test_split, KFold

quick = False
max_files_for_quick_processing = 9

def ReadPlaylists(path, num=None):
    playlists = []

    count = 0
    filenames = os.listdir(path)
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for playlist in mpd_slice['playlists']:
                playlists.append(playlist)
                if len(playlists) == num:
                    return playlists

            count += 1
            print(count)
            if quick and count > max_files_for_quick_processing:
                break


    return playlists

def ReadRatings(path):
    users = []
    items = []

    count = 0
    filenames = os.listdir(path)
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for playlist in mpd_slice['playlists']:
                for track in playlist['tracks']:
                    users.append(playlist['pid'])
                    items.append(track['track_uri'])

            count += 1
            print(count)
            if quick and count > max_files_for_quick_processing:
                break

    raw_userID = list(set(users))
    raw_itemID = list(set(items))

    n_users = len(raw_userID)
    n_items = len(raw_itemID)

    userID = pd.Series(range(n_users),index=raw_userID)
    itemID = pd.Series(range(n_items),index=raw_itemID)

    ratings_dict = {'user': users,
                    'item': items}
    ratings = pd.DataFrame(ratings_dict)

    return userID, itemID, ratings

def JaccardIndex(x, y):
    jaccard_index = np.zeros(y.shape[0])

    x_row, x_col = x.nonzero()
    y_row, y_col = y.nonzero()
    sim_row = np.unique(y_row[np.isin(y_col, x_col)])

    for i in sim_row:
        idx1 = np.intersect1d(x_col, y_col[y_row==i])
        idx2 = np.union1d(x_col, y_col[y_row==i])
        jaccard_index[i] = idx1.size / idx2.size if idx2.size != 0 else 0.0

    return jaccard_index

def precision(rec_lists, true_ratings):
    r_precision = 0.0
    for user_i, rated_items in enumerate(true_ratings):
        n = len(rated_items)
        if n > 0:
            r_precision += np.isin(rec_lists[user_i][:n], rated_items).sum() / n

    n_users = user_i + 1
    r_precision = r_precision / n_users

    return r_precision

def UserBasedCF(RM, train, test):
    predicts = np.zeros(RM[test].shape)
    for i, user_i in enumerate(test):
        sim = JaccardIndex(RM[user_i], RM[train])
#        print(i, user)

        # k nearest neighbors
        k = 100
        knn = np.argsort(sim)[::-1][:k]
        sim_users = train[knn]
        predicts[i] = RM[sim_users].transpose().dot(sim[knn])

    # do not predict rated items
    predicts[RM[test].nonzero()] = 0
    rankings = np.argsort(-predicts)

    return rankings

#def ItemBasedCF(RM, train_users, test_users):
##    n_items = user_to_predict.size
#    which_rated = np.nonzero(user_to_predict)[0]
#    item_similarity = sim_matrix[which_rated].A + sim_matrix[:,which_rated].A.T
#    predicts = np.matmul(user_to_predict[which_rated], item_similarity)
#    
##    predicts = np.zeros(n_items)
##    for j in range(n_items):
##        item_similarity = np.squeeze(sim_matrix.getrow(j).A) + \
##                            np.squeeze(sim_matrix.getcol(j).A.T)
##        # k nearest neighbors
##        predicts[j] = np.matmul(item_similarity[which_rated], user_to_predict[which_rated].T)
#    return predicts

def NMF(RM, train, test):
    nu, ni = RM.shape
    nk = 300
    epsilon = 2e-1
    train = True

    idx_r, idx_c = RM.nonzero()
#    RM[idx_r, idx_c] = 1

    W = np.random.randint(0,2,(nu, ni), dtype=np.int8)
    W[idx_r, idx_c] = 1
    W = csr_matrix(W)

    R = W.multiply(RM)
    P = csr_matrix(np.random.random((nu, nk)) * 1)
    Q = csr_matrix(np.random.random((nk, ni)) * 1)

    P_eps = csr_matrix(np.ones((nu, nk))*1e-5)
    Q_eps = csr_matrix(np.ones((nk, ni))*1e-5)

    old_loss = 0.0
    # train the model use alternate least square

    while train:
        # update P matrix
        P_numerator = R.dot(Q.transpose())
        P_denominator = W.multiply(P.dot(Q)).dot(Q.transpose()) + P_eps
        P_mul = P_numerator.multiply(P_denominator.power(-1))
        
        P_new = P.multiply(P_mul)
        cond1 = np.abs(P_new - P).sum()
        P = P_new.copy()

        # update Q matrix
        Q_numerator = P.transpose().dot(R)
        Q_denominator = P.transpose().dot(W.multiply(P.dot(Q))) + Q_eps
        Q_mul = Q_numerator.multiply(Q_denominator.power(-1))

        Q_new = Q.multiply(Q_mul)
        cond2 = np.abs(Q_new - Q).sum()
        Q = Q_new.copy()

        # compute the loss
        loss = np.power(R[idx_r, idx_c] - P.dot(Q)[idx_r, idx_c], 2).sum()
        avg_loss = loss / idx_r.size
        print(avg_loss, cond1, cond2)

        if np.abs(avg_loss) < epsilon:
#        if cond1 < epsilon and cond2 < epsilon:
            break

    # make prediction
    predicts = P[test].dot(Q)

    # do not predict rated items
    predicts[RM[test].nonzero()] = 0
    predicts = predicts.A
#    rankings = np.argsort(predicts)
    rankings = np.argsort(np.abs(np.ones_like(predicts) - predicts))

    return rankings

def cross_validation(userID, itemID, ratings, n_splits):
    r_precision_stats = []
    
    # CROSS VALIDATION
    print('CROSS VALIDATION:')
    kf = KFold(n_splits=n_splits)
    for train, test in kf.split(userID):
        print('\nTRAIN:', train,'\nTEST:', test)

        t0 = time.time()
        # build rating matrix
        RM = lil_matrix((userID.size, itemID.size))

        for uid, user_i in userID.iloc[train].iteritems():
            rated_items = itemID[ratings[ratings['user']==uid].item].tolist()
            for item_j in rated_items:
                RM[user_i,item_j] += 1

        # for each playlist(user) in test set, the first 5 songs are saved, and
        # the others are taken out for test
        takeout_rated = []
        for uid, user_i in userID.iloc[test].iteritems():
            rated_items = itemID[ratings[ratings.user==uid].item].tolist()        
            n_left = 5#len(rated_items)//2
            left_rated = rated_items[:n_left]
            takeout_rated.append(rated_items[n_left:])
            for item_j in left_rated:
                RM[user_i,item_j] += 1

        RM = RM.tocsr()
        
        t1 = time.time()
        # make recommendations
        rankings = UserBasedCF(RM, train, test)
#        rankings = NMF(RM, train, test)

        t2 = time.time()
        # compute R_precision
        r_precision = precision(rankings, takeout_rated)
        print('R_Precision:', r_precision)
#        print(t1-t0, t2-t1)

        r_precision_stats.append(r_precision)

    r_precision_avg = np.mean(r_precision_stats)
    r_precision_std = np.std(r_precision_stats)
    print('\nR_Precision Mean: ', r_precision_avg)
    print('R_Precision Std: ', r_precision_std)

    return r_precision_stats

if __name__ == '__main__':
    path = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == '--quick':
        quick = True
#    playlists = ReadPlaylists(path)
    userID, itemID, ratings = ReadRatings(path)
    cross_validation(userID, itemID, ratings, n_splits=10)
