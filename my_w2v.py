# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:53:49 2018

@author: lihepeng
"""

import os
import io
import sys
import json
import time
import argparse
import random
from collections import Counter, deque

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from tempfile import gettempdir
from tensorflow.contrib.tensorboard.plugins import projector


def read_playlists(path, num_playlists=None):
    file_count = 0
    playlist_count = 0

    filenames = os.listdir(path)
    for filename in sorted(filenames):
        if filename.startswith('mpd.slice.') and filename.endswith('.json'):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath, 'r', encoding='utf-8')
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
        for playlist in mpd_slice['playlists']:
            read_track(playlist)

            playlist_count += 1
            if playlist_count == num_playlists:
                return

        file_count += 1
        print(file_count)
        if file_count > max_files_for_quick_processing:
            break


def read_track(playlist):
    global playlists, dictionary, list_names, reverse_dictionary

    index = []
    for track in playlist['tracks']:
        song = track['track_name'] + ' by ' + track['artist_name'] + ': '\
                 + track['track_uri'][-3:]
        dictionary[song] = dictionary.get(song, len(dictionary))
        reverse_dictionary[dictionary[song]] = track['track_uri']

        index.append(dictionary.get(song))
    playlists.append(index)
    list_names.append(playlist['name'])


def precision(rec_lists, true_ratings):
    r_precision = 0.0
    for user_i, rated_items in enumerate(true_ratings):
        n = len(rated_items)
        if n > 0:
            r_precision += np.isin(rec_lists[user_i][:n], rated_items).sum() / n

    n_users = user_i + 1
    r_precision = r_precision / n_users

    return r_precision


#def generate_batch(batch_size=1):
#
#    batch_inputs = []
#    batch_labels = []
#
#    sample_playlists = random.sample(playlists, batch_size)
#    for playlist in sample_playlists:
#        song = random.choice(playlist)
#        context_songs = [[i] for i in playlist if i != song]
#
#        batch_inputs.extend([song] * len(context_songs))
#        batch_labels.extend(context_songs)
#
#    return batch_inputs, batch_labels


def generate_batch(batch_size=1):

    batch_inputs = []
    batch_labels = []

    span = 25
    sample_playlists = random.sample(playlists, batch_size)
    for sample_playlist in sample_playlists:
        song = random.choice(sample_playlist)
        context_songs = [[i] for i in sample_playlist if i != song]
        song_to_use = random.choices(context_songs, k=span)

        batch_inputs.extend([song] * span)
        batch_labels.extend(song_to_use)

    return batch_inputs, batch_labels


#def generate_lwa_batch(pos_embeds, neg_embeds, trainset, embed_size):
#
#    batch_size = 128
#
#    # define the batch training set
#    batch_pos_rep = np.ndarray(shape=(batch_size,embed_size), dtype=np.float32)
#    batch_neg_rep = np.ndarray(shape=(batch_size,embed_size), dtype=np.float32)
#    batch_song_rep = np.ndarray(shape=(batch_size,embed_size), dtype=np.float32)
#    batch_labels = np.ndarray(shape=(batch_size), dtype=np.float32)
#
#    # fill data into the batch training set
#    for i in range(batch_size):
#        index_playlist = random.choice(trainset)
#        playlist = data[index_playlist]        
#
#        # sample data from positive class if i is odd or negative class if i is even
#        if i % 2:
#            index_track = random.choice(playlist)
#        else:
#            index_track = random.choice(range(len(dictionary)))
#            while index_track in playlist:
#                index_track = random.choice(range(len(dictionary)))
#
#        # inputs and labels
#        batch_pos_rep[i] = pos_embeds[index_playlist]
#        batch_neg_rep[i] = neg_embeds[index_playlist]
#        batch_song_rep[i] = embed[index_track]
#        batch_labels[i] = i % 2.0
#
#    return batch_pos_rep, batch_neg_rep, batch_song_rep, batch_labels
#
#
#def lwa(embed, trainset, testset):
#
#    # shapes
#    embed_size = embed.shape[1]
#
#    # class representative embeddings for each playlist
#    num_pos_tracks, num_neg_tracks = 5, 5
#    pos_embeds = np.ndarray(shape=(len(data),embed_size),dtype=np.float32)
#    neg_embeds = np.ndarray(shape=(len(data),embed_size),dtype=np.float32)
#    for i, playlist in enumerate(data):
#        # find liked(positive) and disliked(negative) tracks of the playlist
#        pos_tracks = playlist[:num_pos_tracks]
#        similarity = np.matmul(embed[pos_tracks[0]], embed.transpose())
#        neg_tracks = similarity.argsort()[:num_neg_tracks]
#
#        # compute embeddings for positive and negative classes
#        pos_embeds[i] = np.sum(embed[pos_tracks],axis=0)/num_pos_tracks
#        neg_embeds[i] = np.sum(embed[neg_tracks],axis=0)/num_neg_tracks
#
#    # define tensor graph
#    graph = tf.Graph()
#    with graph.as_default():
#
#        # Input data
#        with tf.name_scope('inputs'):
#            pos_rep = tf.placeholder(tf.float32, 
#                                          shape=[None, embed_size])
#            neg_rep = tf.placeholder(tf.float32, 
#                                          shape=[None, embed_size])
#            song_rep = tf.placeholder(tf.float32, 
#                                          shape=[None, embed_size])
#            labels = tf.placeholder(tf.float32, shape=[None])
#
#        # Ops and variables pinned to the CPU
#        with tf.device('/cpu:0'):
#            # weights
#            with tf.name_scope('weights'):
#                pos_weights = tf.Variable(
#                    tf.random_uniform([embed_size], -1.0, 1.0))
#                neg_weights = tf.Variable(
#                    tf.random_uniform([embed_size], -1.0, 1.0))
#                song_weights = tf.Variable(
#                    tf.random_uniform([embed_size], -1.0, 1.0))
##                weights = tf.multiply(pos_rep, pos_weights) + \
##                                tf.multiply(neg_rep, neg_weights)
##                print(weights.shape)
#            # biases
#            with tf.name_scope('biases'):
#                biases = tf.Variable(tf.zeros([1]), trainable=True)
#
##            y = tf.reduce_sum(tf.multiply(song_rep, weights), axis=1) + biase
#
#            y = tf.reduce_sum(tf.multiply(pos_rep, pos_weights), axis=1) \
#             + tf.reduce_sum(tf.multiply(neg_rep, neg_weights), axis=1) \
#             + tf.reduce_sum(tf.multiply(song_rep, song_weights), axis=1) \
#             + biases
#
#            logits = y
#
#        # loss
#        with tf.name_scope('loss'):
#            loss = tf.reduce_mean(
#                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, 
#                                                        logits=logits))
#
#        # Construct the SGD optimizer
#        with tf.name_scope('optimizer'):
#            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
#
#        # Predictor
#        with tf.name_scope('predictor'):
#            predictor = tf.sigmoid(logits)
#
#        # Add variable initializer.
#        init = tf.global_variables_initializer()
#
#        # Create a saver.
#        saver = tf.train.Saver()
#
#
#    with tf.Session(graph=graph) as session:
#        # We must initialize all variables before we use them.
#        init.run()
#        print('Initialized')
#
#        # Begin training.
#        num_steps = 20001
#        average_loss = 0
#
#        for step in range(num_steps):
#            batch_set = generate_lwa_batch(pos_embeds, 
#                                           neg_embeds, 
#                                           trainset, 
#                                           embed_size)
#
#            feed_dict = {pos_rep: batch_set[0], 
#                         neg_rep: batch_set[1],
#                         song_rep: batch_set[2], 
#                         labels: batch_set[3]}
#
#            _, loss_val = session.run([optimizer, loss], 
#                                      feed_dict=feed_dict)
#            average_loss += loss_val
#
#            if step % 10000 == 0:
#                if step > 0:
#                    average_loss /= 10000
#                print('Average loss at step ', step, ': ', average_loss)
#                average_loss = 0
#
#        # Save the model for checkpoints.
#        saver.save(session, 'meta_net\model.ckpt')
#
#        # test
#        rankings = []
#        true_ratings = []
#        for index_playlist in testset:
#            batch_pos_rep = np.repeat(pos_embeds[index_playlist][None,:],
#                                      len(dictionary), axis=0)
#            batch_neg_rep = np.repeat(neg_embeds[index_playlist][None,:],
#                                      len(dictionary), axis=0)
#            batch_song_rep = embed
#
#            feed_dict = {pos_rep: batch_pos_rep, 
#                         neg_rep: batch_neg_rep,
#                         song_rep: batch_song_rep}
#
#
#            left_rated = data[index_playlist][:num_pos_tracks]
#            takeout_rated = data[index_playlist][num_pos_tracks:]
#
#            predicts = session.run(predictor, feed_dict=feed_dict)
#            ranking = (-np.squeeze(predicts)).argsort()
#            np.delete(ranking, np.isin(ranking, left_rated))
#
#            rankings.append(ranking)
#            true_ratings.append(takeout_rated)
#
#        # compute R_precision
#        r_precision = precision(rankings, true_ratings)
#        print('R_Precision:', r_precision)
#
#    return r_precision


def knn(embeddings, test):

#    random.seed(896341)

    # class representative embeddings for each playlist
    num_pos_tracks = 5

    # test
    rec_lists = []
    true_lists = []
    for c, index_playlist in enumerate(test):
        left_rated = playlists[index_playlist][:num_pos_tracks]
        takeout_rated = playlists[index_playlist][num_pos_tracks:]
        
        ranking = Counter()
#        prob_val = np.matmul(embeddings[left_rated], embeddings.T)
        prob_val = np.matmul(np.mean(embeddings[left_rated],axis=0)[None,:], embeddings.T)

        sim_songs = np.argsort(-prob_val, axis=1)[:,:600]
        for i in range(sim_songs.shape[0]):
            for j in range(sim_songs.shape[1]):
                if sim_songs[i,j] not in left_rated:
                    ranking[sim_songs[i,j]] += prob_val[i,sim_songs[i,j]]


#        for similarity in prob_val:
#            for j in range(600):
#                idx_song = np.argmax(similarity)
#                if idx_song not in left_rated:
#                    ranking[idx_song] += similarity[idx_song]
#                similarity[idx_song] = 0.0

        predict = dict(ranking.most_common(500))
        rec_lists.append(list(predict.keys()))
        true_lists.append(takeout_rated)
        print(c, index_playlist)
    
    # compute R_precision
    r_precision = precision(rec_lists, true_lists) 
    print('R_Precision:', r_precision)

    return r_precision


if __name__ == '__main__':
    file_path = r'C:\Users\lihepeng\Documents\GitHub\RecSys2018'
    mpd_path = '\mpd_v1\data'
    challenge_path = '\challenge_v1'
 
    if 'data.json' in os.listdir(file_path):
        f = open(file_path+'\data.json', 'r')
        js = f.read()
        f.close()
        data = json.loads(js)
        playlists = data['playlists']
        dictionary = data['dictionary']
        reverse_dictionary = data['reverse_dictionary']
        list_names = data['list_names']
    else:
        max_files_for_quick_processing = 999

        playlists, list_names  = [], []
        dictionary, reverse_dictionary = dict(), dict()
        read_playlists(file_path+mpd_path, num_playlists=None)

        f = open(file_path+'\data.json', 'w+')
        data = {'playlists': playlists, 
                'dictionary': dictionary, 
                'reverse_dictionary': reverse_dictionary, 
                'list_names': list_names}
    
        json.dump(data, f)
        f.close()


#    embeddings = song2vec(embedding_size = 50, num_steps=100001)

    ''' dimensions and parameters '''
    vocabulary_size = len(dictionary)
    embedding_size = 20
    num_steps = 100000000000000
    num_sampled = 1

    valid_size = 5
    valid_window = 500
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    show_embeddings = False

    ''' tensor graph '''
    graph = tf.Graph()
    with graph.as_default():

        # Input data
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # weights & biases
        with tf.device('/cpu:0'):
            # embeddings
            init_embeds = tf.random_uniform(
                            [vocabulary_size, embedding_size], -1.0, 1.0)
            embeddings = tf.get_variable('embeddings', initializer=init_embeds)
            embeds = tf.nn.embedding_lookup(embeddings, train_inputs)

            # weights and biases
            init_weights = tf.truncated_normal(
                            [vocabulary_size, embedding_size], 
                            stddev=1.0 / (embedding_size ** 0.5))
            init_biases = tf.zeros([vocabulary_size])
            weights = tf.get_variable('weights', initializer=init_weights)
            biases = tf.get_variable('biases', initializer=init_biases)

        # loss
        loss = tf.reduce_mean(
                tf.nn.nce_loss(
                        weights=weights,
                        biases=biases,
                        labels=train_labels,
                        inputs=embeds,
                        num_sampled=num_sampled,
                        num_classes=vocabulary_size))

        logits = tf.matmul(embeds, tf.transpose(weights))
        logits = tf.nn.bias_add(logits, biases)
        prob = tf.nn.softmax(logits)

        # optimizertea
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                  valid_dataset)
        similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer
        init = tf.global_variables_initializer()

        # Create a saver
        saver = tf.train.Saver()

    '''' Begin training '''
#    with tf.Session(graph=graph) as session:
    sess = tf.Session(graph=graph)

    model_path = r'C:\Users\lihepeng\Documents\GitHub\RecSys2018\model'
    model_name = '\song2_vec.ckpt'
    if model_name[1:]+'.index' in os.listdir(model_path):
        # Restore variables from disk.
        saver.restore(sess, model_path + model_name)
    else:
        # initialization
        sess.run(init)
        print('Initialized')

    # train
    global buffer
    buffer = deque()

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(32)
        feed_dict = {train_inputs:batch_inputs, train_labels:batch_labels}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        # print loss results every 1000 steps
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000

            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 1000000 == 0:
#            sim = sess.run(similarity)
#            for i in range(valid_size):
#                valid_word = valid_examples[i]
#                top_k = 8  # number of nearest neighbors
#                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
#                log_str =  'Nearest to %s:' % valid_word
#                for k in range(top_k):
#                    close_word = nearest[k]
#                    log_str = '%s %s,' % (log_str, close_word)
#                print(log_str,'\n')

            final_embeddings = sess.run(normalized_embeddings)
            saver.save(sess, model_path + model_name)

            # test
            test = random.sample(range(len(playlists)), 100)
            knn(final_embeddings, test)

#    # cross validataion
#    r_precision_stats = []
#
#    n_splits = 10
#    kf = KFold(n_splits=n_splits)
#    for train, test in kf.split(playlists):
#        r_precision = knn(final_embeddings, test)
#        r_precision_stats.append(r_precision)
#
#    r_precision_avg = np.mean(r_precision_stats)
#    r_precision_std = np.std(r_precision_stats)
#    print('R_Precision Mean: ', r_precision_avg)
#    print('R_Precision Std: ', r_precision_std)

if show_embeddings:
    def plot_with_labels(low_dim_embs, labels, filename):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches 
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(
                    label, 
                    xy=(x, y), 
                    xytext=(5, 2), 
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
        plt.savefig(filename)
    
    try:
        # pylint: disable=g-import-not-at-top
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        from tempfile import gettempdir
        tsne = TSNE(
            perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [i for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))
    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)



