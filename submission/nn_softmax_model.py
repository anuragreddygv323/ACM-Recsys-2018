# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:32:56 2018

@author: lihepeng
"""

import os
import re
import json
import random
from collections import Counter

import numpy as np
import tensorflow as tf

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


file_path = r'C:\Users\lihepeng\Documents\GitHub\RecSys2018'
mpd_path = '\mpd_v1\data'
challenge_path = '\challenge_v1'

embed_path = r'C:\Users\lihepeng\Documents\GitHub\RecSys2018\embeds.npy'
song_embeds = np.load(embed_path)
    
if 'data.json' in os.listdir(file_path):
    f = open(file_path+'\data.json', 'r')
    js = f.read()
    f.close()
    data = json.loads(js)
    playlists = data['playlists']
    dictionary = data['dictionary']
    reverse_dictionary = data['reverse_dictionary']
    list_names = data['list_names']
    name_embeds = data['name_embeds']

else:
    max_files_for_quick_processing = 999

    playlists, list_names  = [], []
    dictionary, reverse_dictionary = dict(), dict()
    read_playlists(file_path+mpd_path, num_playlists=None)

    dict_names = dict()
    pattern = '[\s\[\]\{\}\(\)\:\;\'\"\?\<\>\,\.\/\~\`\!\@\#\$\%\^\*\_\-\+\=\|\\\\]'
    for i in range(len(playlists)):
        name = re.sub(pattern, '', list_names[i]).lower()
        list_names[i] = name
        track_list = dict_names.get(name, set(playlists[i]))
        dict_names[name] = track_list.union(set(playlists[i]))
        if i % 2000 == 0:
            print(i)

    name_embeds=dict()
    for name, tracks in dict_names.items():
        name_embed = np.sum(song_embeds[list(tracks)], axis=0)
        norm_name_embed = name_embed/np.linalg.norm(name_embed)
        name_embeds[name] = norm_name_embed.tolist()
    
    del dict_names

    f = open(file_path+'\data.json', 'w+')
    data = {'playlists': playlists, 
            'dictionary': dictionary, 
            'reverse_dictionary': reverse_dictionary, 
            'list_names': list_names,
            'name_embeds': name_embeds}

    json.dump(data, f)
    f.close()


def precision(rec_lists, true_ratings):
    r_precision = 0.0
    for user_i, rated_items in enumerate(true_ratings):
        n = len(rated_items)
        if n > 0:
            r_precision += np.isin(rec_lists[user_i][:n], rated_items).sum() / n

    n_users = user_i + 1
    r_precision = r_precision / n_users

    return r_precision


def generate_batch(batch=1):

    batch_name_embeds = []
    batch_seed_embeds = []
    batch_target_embeds = []
    batch_labels = []

    for b in range(batch):
        # randomly choose a playlist
        i = random.choice(range(len(playlists)))

        # get the playlist's name_embeds
        norm_name_embeds = name_embeds[list_names[i]]

        # get the playlists's seeds_embeds
        seeds = random.sample(playlists[i], 5)
        seed_embeds = np.sum(song_embeds[seeds], axis=0)
        norm_seed_embeds = seed_embeds/np.linalg.norm(seed_embeds)
        
        # get a target track and its embeds, and the label
        if b % 2:
            target = random.choice(playlists[i])
            norm_target_embeds = song_embeds[target]
            label = [1, 0]
        else:
            target = random.choice(range(song_embeds.shape[0]))
            norm_target_embeds = song_embeds[target]
            label = [0, 1] if target not in playlists[i] else [1, 0]

        batch_name_embeds.append(norm_name_embeds)
        batch_seed_embeds.append(norm_seed_embeds)
        batch_target_embeds.append(norm_target_embeds)
        batch_labels.append(label)

    return batch_name_embeds, batch_seed_embeds, batch_target_embeds, batch_labels


''' NN softmax model '''
embed_size = 100
hidden_node_size = [200, 200]
vocabulary_size = 2262292
num_sampled = 32

graph = tf.Graph()
with graph.as_default():

    # inputs & outputs
    train_name_embeds = tf.placeholder(tf.float32, shape=[None, embed_size])
    train_seed_embeds = tf.placeholder(tf.float32, shape=[None, embed_size])
    train_target_embeds = tf.placeholder(tf.float32, shape=[None, embed_size])
    train_labels = tf.placeholder(tf.float32, shape=[None, 2])

    cross_feature_1 = tf.multiply(train_name_embeds, train_seed_embeds)
    cross_feature_2 = tf.multiply(train_name_embeds, train_target_embeds)
    cross_feature_3 = tf.multiply(train_seed_embeds, train_target_embeds)

    feature = tf.concat([train_name_embeds,
                          train_seed_embeds,
                          train_target_embeds,
                          cross_feature_1,
                          cross_feature_2,
                          cross_feature_3], 1)

    # variables
    W1 = tf.get_variable('W1', initializer=tf.random_uniform(
            [hidden_node_size[0], 6*embed_size], -1.0, 1.0))
    b1 = tf.get_variable('b1', initializer=tf.ones([hidden_node_size[0]]))

    W2 = tf.get_variable('W2', initializer=tf.random_uniform(
            [hidden_node_size[1], hidden_node_size[0]], -1.0, 1.0))
    b2 = tf.get_variable('b2', initializer=tf.ones([hidden_node_size[1]]))

    W_softmaxt = tf.get_variable('W_softmaxt', initializer=tf.random_uniform(
            [2, hidden_node_size[1]], -1.0, 1.0))
    b_softmax = tf.get_variable('b_softmax', initializer=tf.ones(
            [2]))

    # model
    h1 = tf.nn.sigmoid(tf.matmul(feature, tf.transpose(W1)) + b1)
    h2 = tf.nn.sigmoid(tf.matmul(h1, tf.transpose(W2)) + b2)
    logits = tf.matmul(h2, tf.transpose(W_softmaxt)) + b_softmax
    prob = tf.nn.softmax(logits)

    # loss
    loss = tf.losses.softmax_cross_entropy(train_labels, logits)

    # optimizertea
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # Add variable initializer
    init = tf.global_variables_initializer()

    # Create a saver
    saver = tf.train.Saver()

'''' Begin training '''
#    with tf.Session(graph=graph) as session:
sess = tf.Session(graph=graph)

model_path = r'C:\Users\lihepeng\Documents\GitHub\RecSys2018\challenge_v1\model'
model_name = r'\nn_softmax.ckpt'
if model_name[1:]+'.index' in os.listdir(model_path):
    # Restore variables from disk.
    saver.restore(sess, model_path + model_name)
else:
    # initialization
    sess.run(init)
    print('Initialized')

num_steps = 100000000000000000
batch_size = 32

# train
average_loss = 0
for step in range(num_steps):
    batch_name_embeds, batch_seed_embeds, \
    batch_target_embeds, batch_labels = generate_batch(batch_size)

    feed_dict = {train_name_embeds:batch_name_embeds, 
                 train_seed_embeds:batch_seed_embeds,
                 train_target_embeds:batch_target_embeds,
                 train_labels:batch_labels}
    _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    # print loss results every 1000 steps
    if step % 2000 == 0:
        if step > 0:
            average_loss /= 2000

        print('Average loss at step ', step, ': ', average_loss)
        average_loss = 0

    # test
    if step % 100000 == 0:
        saver.save(sess, model_path + model_name)

        test = random.sample(range(len(playlists)), 10)

        # class representative embeddings for each playlist
        num_pos_tracks = 5

        # test
        rec_lists = []
        true_lists = []
        for c, index_playlist in enumerate(test):
            left_rated = playlists[index_playlist][:num_pos_tracks]
            takeout_rated = playlists[index_playlist][num_pos_tracks:]

            # get the playlist's name_embeds
            norm_name_embeds = name_embeds[list_names[index_playlist]]

            # get the playlists's seeds_embeds
            seeds = left_rated
            seed_embeds = np.sum(song_embeds[seeds], axis=0)
            norm_seed_embeds = seed_embeds/np.linalg.norm(seed_embeds)

            # get the target tracks and their embeds
            a = (norm_seed_embeds + norm_name_embeds)/2
            similarity = np.matmul(a, song_embeds.T)
            sim_song_idxs = np.argsort(-similarity)[:10000]
            norm_target_embeds = song_embeds[sim_song_idxs]

            batch_name_embeds = [norm_name_embeds] * len(norm_target_embeds)
            batch_seed_embeds = [norm_seed_embeds] * len(norm_target_embeds)
            batch_target_embeds = norm_target_embeds

            # caompute probability
            feed_dict = {train_name_embeds:batch_name_embeds, 
                         train_seed_embeds:batch_seed_embeds,
                         train_target_embeds:batch_target_embeds}
            p = sess.run(prob, feed_dict=feed_dict)

            candidates = sim_song_idxs[np.argsort(-p[:,0])]
            reccommends = list(set(candidates) - set(seeds))[:500]

            rec_lists.append(reccommends)
            true_lists.append(takeout_rated)

            print(c)    

        r_precision = precision(rec_lists, true_lists)
        print('R_Precision:', r_precision)
















