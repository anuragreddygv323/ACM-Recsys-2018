# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:54:02 2018

@author: lihepeng
"""

import sys
import json
import csv
import numpy as np
from collections import Counter, deque

NTRACKS = 500

challenge_path = r'C:\Users\lihepeng\Documents\GitHub\RecSys2018\challenge_v1\challenge_set.json'
submission_path = r'C:\Users\lihepeng\Documents\GitHub\RecSys2018\challenge_v1\sample_submission.csv'
data_path = r'C:\Users\lihepeng\Documents\GitHub\RecSys2018\data.json'
embed_path = r'C:\Users\lihepeng\Documents\GitHub\RecSys2018\embeds_0517.npy'

has_team_info = False

# read challenge data
f = open(challenge_path)
js = f.read()
f.close()
challenge = json.loads(js)

# read embeddings
embeds = np.load(embed_path)

# read data
f = open(data_path, 'r')
js = f.read()
f.close()
data = json.loads(js)
playlists = data['playlists']
dictionary = data['dictionary']
reverse_dictionary = data['reverse_dictionary']
list_names = data['list_names']

del data

name_embeds = []
for i, playlist in enumerate(playlists):
    list_name_embeding = np.mean(embeds[playlist],axis=0)
    name_embeds.append(list_name_embeding)

seed_embeds = dict()
for i, playlist in enumerate(challenge['playlists']):
    playlist_embed = []
    if len(playlist['tracks']) == 0:
        if playlist['name'] in list_names:
            for idx, x in enumerate(list_names):
                if x == playlist['name']:
                    playlist_embed.append(name_embeds[idx])
            playlist_embed = [np.mean(playlist_embed, axis=0)]
        else:
            sim = []
            y = set(playlist['name'])
            for idx, x in enumerate(list_names):
                x = set(x)
                sim.append(len(y.intersection(x))/len(y.union(x)))
            sim = np.array(sim,dtype=np.float32)
            idx = np.argsort(-sim)[:5].tolist()
            for j in idx:
                playlist_embed.append(name_embeds[j])

    elif len(playlist['tracks']) == 1:
        track = playlist['tracks'][0]
        track_uris = track['track_name'] + ' by ' + track['artist_name'] \
                        + ': ' + track['track_uri'][-3:]
        playlist_embed.append(embeds[dictionary[track_uris]])
        if playlist['name'] in list_names:
            for idx, x in enumerate(list_names):
                if x == playlist['name']:
                    playlist_embed.append(name_embeds[idx])
        else:
            sim = []
            y = set(playlist['name'])
            for idx, x in enumerate(list_names):
                x = set(x)
                sim.append(len(y.intersection(x))/len(y.union(x)))
            sim = np.array(sim,dtype=np.float32)
            idx = np.argsort(-sim)[:5].tolist()
            for j in idx:
                playlist_embed.append(name_embeds[j])

    else:
        for track in playlist['tracks']:
            track_uris = track['track_name'] + ' by ' + track['artist_name'] \
                             + ': ' + track['track_uri'][-3:]
            playlist_embed.append(embeds[dictionary[track_uris]])

    seed_embeds[playlist['pid']] = np.array(playlist_embed)
    print(i)

del playlists, dictionary, list_names, name_embeds, sim

# generate reccomend list
csvfile = open('submission.csv', 'w', newline='')
writer = csv.writer(csvfile, delimiter=',', 
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

team_info = ['team_info', 'main', 'CISA', 'hepengli@uri.edu']
writer.writerow(team_info)
writer.writerow([])

count = 0
for key, value in seed_embeds.items():
    count += 1

    value1 = np.mean(value,axis=0)[None,:]#value[:10]#
    prob_val = np.matmul(value1, embeds.T)
    ranking = Counter()

#        for similarity in prob_val:
#            for _ in range(NTRACKS):
#                idx_song = np.argmax(similarity)
#                if embeds[idx_song] not in value:
#                    ranking[idx_song] += similarity[idx_song]
#                similarity[idx_song] = 0.0

    sim_songs = np.argsort(-prob_val, axis=1)[:,:1000]
    for i in range(sim_songs.shape[0]):
        for j in range(sim_songs.shape[1]):
            if embeds[sim_songs[i,j]] not in value:
                ranking[sim_songs[i,j]] += prob_val[i,sim_songs[i,j]]

    predict = dict(ranking.most_common(NTRACKS))
    rec_list = [reverse_dictionary[str(index)] for index in predict.keys()]

    writer.writerow([key] + rec_list)
    print(count, key)

csvfile.close()

import gzip
import shutil
with open(r'C:\Users\lihepeng\Documents\GitHub\RecSys2018\challenge_v1\submission.csv', 'rb') as submission_csv:
    with gzip.open(r'C:\Users\lihepeng\Documents\GitHub\RecSys2018\challenge_v1\submission.csv.gz', 'wb') as submission_csv_gz:
        shutil.copyfileobj(submission_csv, submission_csv_gz)

