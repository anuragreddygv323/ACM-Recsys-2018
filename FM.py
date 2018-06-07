# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 22:21:01 2018

@author: lihepeng
"""

import os
import sys
import json
from collections import Counter
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold


def read_playlists(path, num_playlists=None):
  global data, dict_track, dict_artist, dict_album, dict_span, playlists
  all_tracks = dict()

  file_count = 0
  playlist_count = 0

  filenames = os.listdir(path)
  for filename in sorted(filenames):
    if filename.startswith("mpd.slice.") and filename.endswith(".json"):
      fullpath = os.sep.join((path, filename))
      f = open(fullpath, 'r', encoding='utf-8')
      js = f.read()
      f.close()
      mpd_slice = json.loads(js)
    for playlist in mpd_slice['playlists']:
      sim_tracks = []
      for track_i in playlist['tracks']:
        track = track_i['track_name']
        artist = track_i['artist_name']
        album = track_i['album_name']
        span = track_i['duration_ms']
        dict_track[track] = dict_track.get(track, len(dict_track))
        dict_artist[artist] = dict_artist.get(artist, len(dict_artist))
        dict_album[album] = dict_album.get(album, len(dict_album))
        dict_span[span] = dict_span.get(span, len(dict_span))

        uri = track_i['track_uri'][14:]
        all_tracks[uri] = all_tracks.get(uri, 
                          [len(all_tracks), 
                           dict_track[track], 
                           dict_artist[artist], 
                           dict_album[album], 
                           dict_span[span]])
        sim_tracks.append(all_tracks[uri][0])

      for track_i in playlist['tracks']:
        uri = track_i['track_uri'][14:]
        idx = all_tracks[uri][0]
        neighbors = set(sim_tracks)
        neighbors.remove(idx)
        if not data.get(idx):
          data[idx] = (all_tracks[uri][1:], neighbors)
        else:
          data[idx][1].union(neighbors)

      playlists.append(sim_tracks)
      playlist_count += 1
      if playlist_count == num_playlists:
        return

    file_count += 1
    print(file_count)
    if file_count > max_files_for_quick_processing:
      break


def precision(rec_lists, true_ratings):
    r_precision = 0.0
    for user_i, rated_items in enumerate(true_ratings):
        n = len(rated_items)
        if n > 0:
            r_precision += np.isin(rec_lists[user_i][:n], rated_items).sum() / n

    n_users = user_i + 1
    r_precision = r_precision / n_users

    return r_precision


def generate_batch(batch_size, input_size):
  global data, dict_track, dict_artist, dict_album, dict_span
  num_tracks = len(data)

  batch_inputs = np.ndarray(shape=[batch_size, input_size], dtype=np.int32)
  batch_lables = np.ndarray(shape=[batch_size, 1], dtype=np.int32)

  for i in range(batch_size):
    track_0 = np.random.choice(range(num_tracks))
    similar_tracks = list(data[track_0][1])
    dissimilar_tracks = list(set(data.keys()) - data[track_0][1])
    if np.random.rand() > 0.5:
      track_1 = np.random.choice(similar_tracks)
      batch_inputs[i] = data[track_0][0] + data[track_1][0]
      batch_lables[i] = [1.0]
    else:
      track_1 = np.random.choice(dissimilar_tracks)
      batch_inputs[i] = data[track_0][0] + data[track_1][0]
      batch_lables[i] = [0.0]

  return batch_inputs, batch_lables

path = r'C:\Users\lihepeng\Documents\GitHub\RecSys2018\mpd_v1\data'
max_files_for_quick_processing = 0

data, playlists = dict(), []
dict_track, dict_artist, dict_album, dict_span = dict(), dict(), dict(), dict()
read_playlists(path, num_playlists=None)


''' dimensions and sizes '''
input_size = 8
embed_size = 8
batch_size = 32
num_steps = 200001

track_size = len(dict_track)
artist_size = len(dict_artist)
album_size = len(dict_album)
span_size = len(dict_span)

lr = 1.0

''' build the graph '''
graph = tf.Graph()
with graph.as_default():

  # inputs
  inputs = tf.placeholder(dtype=tf.int32, shape=[None, input_size])
  labels = tf.placeholder(dtype=tf.float32, shape=[None, 1])

  # embeddings & weights & biases
  with tf.device('/cpu:0'):
    # track embeddings
    track_embed = tf.get_variable(
      name='track_embed',
      dtype=tf.float32,
      initializer=tf.random_uniform([track_size, embed_size], -1.0, 1.0))
    t0 = tf.nn.embedding_lookup(track_embed, inputs[:,0])
    t1 = tf.nn.embedding_lookup(track_embed, inputs[:,4])

    # artist embeddings
    artist_embed = tf.get_variable(
      name='artist_embed',
      dtype=tf.float32,
      initializer=tf.random_uniform([artist_size, embed_size], -1.0, 1.0))
    a0 = tf.nn.embedding_lookup(artist_embed, inputs[:,1])
    a1 = tf.nn.embedding_lookup(artist_embed, inputs[:,5])

    # alnum embeddings
    album_embed = tf.get_variable(
      name='album_embed',
      dtype=tf.float32,
      initializer=tf.random_uniform([album_size,embed_size], -1.0, 1.0))
    l0 = tf.nn.embedding_lookup(album_embed, inputs[:,2])
    l1 = tf.nn.embedding_lookup(album_embed, inputs[:,6])

    # span embeddings
    span_embed = tf.get_variable(
      name='length_embed',
      dtype=tf.float32,
      initializer=tf.random_uniform([span_size,embed_size], -1.0, 1.0))
    d0 = tf.nn.embedding_lookup(span_embed, inputs[:,3])
    d1 = tf.nn.embedding_lookup(span_embed, inputs[:,7])

    # weights & bias
    weights = tf.get_variable(
      name='weights',
      dtype=tf.float32,
      initializer=tf.truncated_normal(
          shape=[embed_size * input_size, 1], stddev=1.0), 
      trainable=False)
    bias = tf.get_variable(
            name='bias', 
            dtype=tf.float32, 
            initializer=1.0, 
            trainable=False)

  # loss
  feature_0 = tf.concat([t0, a0, l0, d0], axis=1)
  feature_1 = tf.concat([t1, a1, l1, d1], axis=1)
  features = tf.concat([t0, a0, l0, d0, t1, a1, l1, d1], axis=1)
  logits = tf.matmul(features, weights) + \
      tf.reduce_sum(tf.multiply(feature_0, feature_1), axis=1)[:,None] * bias
  prob = tf.nn.sigmoid(logits)

  regularizer = tf.nn.l2_loss(track_embed) + tf.nn.l2_loss(artist_embed) + \
                  tf.nn.l2_loss(album_embed) + tf.nn.l2_loss(span_embed)
  loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              labels=labels, logits=logits))# + 0.001 * regularizer

  optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

  # Add variable initializer.
  init = tf.global_variables_initializer()

''' Begin training '''
with tf.Session(graph=graph) as session:

  # initialization
  init.run()
  print('Initialized')

  # train
  average_loss = 0
  for step in range(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size, input_size)
    feed_dict = {inputs:batch_inputs, labels: batch_labels}
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    # print loss results every 1000 steps
    if step % 200 == 0:
      if step > 0:
        average_loss /= 200
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

  track_embeddings = track_embed.eval()
  artist_embeddings = artist_embed.eval()
  album_embeddings = album_embed.eval()
  span_embeddings = span_embed.eval()

  # test / cross validataion
  num_pos_tracks, num_neg_tracks = 5, 5
  r_precision_stats = []

  n_splits = 10
  kf = KFold(n_splits=n_splits)
  for train, test in kf.split(playlists):
    print('\nTRAIN:', train,'\nTEST:', test)

    rec_lists, true_lists = [], []
    for index_playlist in test:
      left_rated = playlists[index_playlist][:num_pos_tracks]
      takeout_rated = playlists[index_playlist][num_pos_tracks:]
      to_eval = set(data.keys()) - set(left_rated)

      prob_val = 0.0
      for track_0 in left_rated:
        test_inputs = []
        for track_1 in to_eval:
          test_inputs.append(data[track_0][0] + data[track_1][0])
        prob_val = session.run(prob, feed_dict={inputs: test_inputs})

      recommend = np.argsort(-np.squeeze(prob_val))[:500]
      rec_lists.append(recommend)
      true_lists.append(takeout_rated)

    # compute R_precision
    r_precision = precision(rec_lists, true_lists)
    print('R_Precision:', r_precision)

    r_precision_stats.append(r_precision)

r_precision_avg = np.mean(r_precision_stats)
r_precision_std = np.std(r_precision_stats)
print('R_Precision Mean: ', r_precision_avg)
print('R_Precision Std: ', r_precision_std)