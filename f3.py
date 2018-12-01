import utils
import f2

import numpy as np

index_i = f2.index_i
lemma_i = f2.lemma_i
pos_i = f2.pos_i
morpho_i = f2.morpho_i
governor_i = f2.governor_i
label_i = f2.label_i
pred_pos_i = 6
next_pos_i = 7

def add_neighbors_pos(sentences):
    results = []
    for i, s in enumerate(sentences):
        last_pos = ''
        pred_poss = []
        next_poss = []
        for j in range(len(s) - 1):
            pred_poss.append(last_pos)
            next_poss.append(s[j+1][f2.pos_i])
            last_pos = s[j][f2.pos_i]
        pred_poss.append(last_pos)
        next_poss.append('')
        results.append(np.concatenate(
            (
                s, np.array(pred_poss).reshape(-1, 1), 
                np.array(next_poss).reshape(-1, 1)
            ), 
            axis=1
        ))
    return results

sentences_fr_train = add_neighbors_pos(f2.sentences_fr_train)
sentences_fr_dev = add_neighbors_pos(f2.sentences_fr_dev)
sentences_fr_test = add_neighbors_pos(f2.sentences_fr_test)

def create_example(w1, w2, *args, positive=True):
    morphos_vec = args[0]
    embeddings = args[1]
    mean_embedding = args[2]
    x_splitted = args[3]

    dist = int(w2[f2.index_i]) - int(w1[f2.index_i])

    pos1 = np.zeros(len(utils.pos_2_1hot))
    pos1[utils.pos_2_1hot[w1[f2.pos_i]]] = 1

    pos1_pred = np.zeros(len(utils.pos_2_1hot))
    if w1[pred_pos_i] != '':
        pos1_pred[utils.pos_2_1hot[w1[pred_pos_i]]] = 1

    pos1_next = np.zeros(len(utils.pos_2_1hot))
    if w1[next_pos_i] != '':
        pos1_next[utils.pos_2_1hot[w1[next_pos_i]]] = 1

    pos2 = np.zeros(len(utils.pos_2_1hot))
    pos2[utils.pos_2_1hot[w2[f2.pos_i]]] = 1

    pos2_pred = np.zeros(len(utils.pos_2_1hot))
    if w2[pred_pos_i] != '':
        pos2_pred[utils.pos_2_1hot[w2[pred_pos_i]]] = 1

    pos2_next = np.zeros(len(utils.pos_2_1hot))
    if w2[next_pos_i] != '':
        pos2_next[utils.pos_2_1hot[w2[next_pos_i]]] = 1
    
    morpho1 = f2.convert_morpho(w1, morphos_vec)
    morpho2 = f2.convert_morpho(w2, morphos_vec)
    
    if w1[f2.lemma_i] in embeddings:
        embedding1 = embeddings[w1[f2.lemma_i]]
    else:
        embedding1 = mean_embedding
    if w2[f2.lemma_i] in embeddings:
        embedding2 = embeddings[w2[f2.lemma_i]]
    else:
        embedding2 = mean_embedding
    
    x = np.concatenate(([dist], morpho1, morpho2, embedding1, embedding2, pos1_pred, pos1, pos1_next, pos2_pred, pos2, pos2_next))
    label = np.zeros(37)
    
    y = [0, 0]
    if positive:
        if w1[f2.governor_i] == w2[f2.index_i]:
            g, d = w2, w1
            y[0] = 1
        else:
            d, g = w2, w1
            y[1] = 1
        l = d[label_i].split(':', 1)[0]
        label[utils.labels_2_1hot[l]] = 1

    if x_splitted:
        x = x.reshape(1, -1)
        x = [x[:, :757], x[:, 757:811], x[:, 811:]]

    return x, np.concatenate((y, label))