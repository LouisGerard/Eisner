import utils
import numpy as np
from collections import Counter
import re
from scipy.spatial.distance import cosine

index_i = 0
lemma_i = 1
pos_i = 2
morpho_i = 3
governor_i = 4
label_i = 5

sentences_fr_train = utils.read_conllu("UD_French-GSD/fr_gsd-ud-train.conllu", features_enabled=[0, 2, 3, 5, 6, 7], root=[0, 'ROOT', 'ROOT', '_', 0, 'root'])
sentences_fr_dev = utils.read_conllu("UD_French-GSD/fr_gsd-ud-dev.conllu", features_enabled=[0, 2, 3, 5, 6, 7], root=[0, 'ROOT', 'ROOT', '_', 0, 'root'])
sentences_fr_test = utils.read_conllu("UD_French-GSD/fr_gsd-ud-test.conllu", features_enabled=[0, 2, 3, 5, 6, 7], root=[0, 'ROOT', 'ROOT', '_', 0, 'root'])

sentences_en_train = utils.read_conllu("UD_English-LinES/en_lines-ud-train.conllu", features_enabled=[0, 2, 3, 5, 6, 7], root=[0, 'ROOT', 'ROOT', '_', 0, 'root'])
sentences_en_dev = utils.read_conllu("UD_English-LinES/en_lines-ud-dev.conllu", features_enabled=[0, 2, 3, 5, 6, 7], root=[0, 'ROOT', 'ROOT', '_', 0, 'root'])
sentences_en_test = utils.read_conllu("UD_English-LinES/en_lines-ud-test.conllu", features_enabled=[0, 2, 3, 5, 6, 7], root=[0, 'ROOT', 'ROOT', '_', 0, 'root'])

sentences_nl_train = utils.read_conllu("UD_Dutch-LassySmall/nl_lassysmall-ud-train.conllu", features_enabled=[0, 2, 3, 5, 6, 7], root=[0, 'ROOT', 'ROOT', '_', 0, 'root'])
sentences_nl_dev = utils.read_conllu("UD_Dutch-LassySmall/nl_lassysmall-ud-dev.conllu", features_enabled=[0, 2, 3, 5, 6, 7], root=[0, 'ROOT', 'ROOT', '_', 0, 'root'])
sentences_nl_test = utils.read_conllu("UD_Dutch-LassySmall/nl_lassysmall-ud-test.conllu", features_enabled=[0, 2, 3, 5, 6, 7], root=[0, 'ROOT', 'ROOT', '_', 0, 'root'])

sentences_ja_train = utils.read_conllu("UD_Japanese-GSD-master/ja_gsd-ud-train.conllu", features_enabled=[0, 2, 3, 5, 6, 7], root=[0, 'ROOT', 'ROOT', '_', 0, 'root'])
sentences_ja_dev = utils.read_conllu("UD_Japanese-GSD-master/ja_gsd-ud-dev.conllu", features_enabled=[0, 2, 3, 5, 6, 7], root=[0, 'ROOT', 'ROOT', '_', 0, 'root'])
sentences_ja_test = utils.read_conllu("UD_Japanese-GSD-master/ja_gsd-ud-test.conllu", features_enabled=[0, 2, 3, 5, 6, 7], root=[0, 'ROOT', 'ROOT', '_', 0, 'root'])

def count_morphos(sentences):
    morphos = {}
    no_morpho_count = 0
    word_count = 0

    for s in sentences:
        for w in s:
            word_count += 1
            if w[morpho_i] == '_':
                no_morpho_count += 1
                continue
            for m in w[morpho_i].split('|'):
                m = m.split('=')
                if m[0] not in morphos:
                    morphos[m[0]] = Counter()
                morphos[m[0]][m[1]] += 1
    return morphos, no_morpho_count, word_count

def remove_morpho(to_rm, sentences, morphos):
    for s in sentences:
        for w in s:
            for m in to_rm:
                if m in w[morpho_i]:
                    # print(w)
                    w[morpho_i] = re.sub(m + '=(.+?)\|', '', w[morpho_i])

    for m in to_rm:
        del morphos[m]

def morpho_2_vec(morphos):
    result = {}
    i = 0
    for k, m in morphos.items():
        result[k] = {}
        keys = m.keys()
        for k2 in m.keys():
            result[k][k2] = i
            i += 1
    return result

def convert_morpho(w, morphos_vec):
    result = np.zeros(78)
    if w[morpho_i] == '_':
        return result
    for m in w[morpho_i].split('|'):
        m = m.split('=')
        if m[0] not in morphos_vec or m[1] not in morphos_vec[m[0]]:
            continue
        result[morphos_vec[m[0]][m[1]]] = 1
    return result

def load_embedding_tsv(filename):
    embeddings = {}

    with open(filename, 'r') as f:
        i = 0
        for line in f:
            i += 1
            if line[0] != ' ':  # new word
                if 'word' in locals():
                    embeddings[word] = np.array(vec, dtype=np.float)
                line = line.split('\t')
                word = line[1]
                vec = line[2].lstrip('[').split()
                continue
            
            vec = vec + line.strip(' ]\n').split()
    return embeddings

def most_similar(word, embeddings, n=10):
    vec = embeddings[word]
    top = []
    top_words = []
    for i in range(n):
        top.append(float('inf'))
        top_words.append('')

    cpt = 0
    for w, v in embeddings.items():
        if w == word:
            continue
        cpt += 1
        dist = cosine(vec, v)
        i = 0
        while i < len(top) and dist < top[i]:
            i += 1
        top = top[:i] + [dist] + top[i:]
        top = top[1:]
        top_words = top_words[:i] + [w] + top_words[i:]
        top_words = top_words[1:]

    return zip(reversed(top_words), reversed(top))

def default_embedding(embeddings, embedding_size):
    mean_embedding = np.zeros(embedding_size)
    for w, v in embeddings.items():
        mean_embedding = mean_embedding + v
    return mean_embedding / len(embeddings)

def create_example(w1, w2, *args, positive=True):
    morphos_vec = args[0]
    embeddings = args[1]
    mean_embedding = args[2]

    dist = int(w2[index_i]) - int(w1[index_i])

    pos1 = np.zeros(len(utils.pos_2_1hot))
    pos1[utils.pos_2_1hot[w1[pos_i]]] = 1

    pos2 = np.zeros(len(utils.pos_2_1hot))
    pos2[utils.pos_2_1hot[w2[pos_i]]] = 1
    
    morpho1 = convert_morpho(w1, morphos_vec)
    morpho2 = convert_morpho(w2, morphos_vec)
    
    if w1[lemma_i] in embeddings:
        embedding1 = embeddings[w1[lemma_i]]
    else:
        embedding1 = mean_embedding
    if w2[lemma_i] in embeddings:
        embedding2 = embeddings[w2[lemma_i]]
    else:
        embedding2 = mean_embedding
    
    x = np.concatenate(([dist], pos1, pos2, morpho1, morpho2, embedding1, embedding2))
    label = np.zeros(37)
    
    y = [0, 0]
    if positive:
        if w1[governor_i] == w2[index_i]:
            g, d = w2, w1
            y[0] = 1
        else:
            d, g = w2, w1
            y[1] = 1
        l = d[label_i].split(':', 1)[0]
        label[utils.labels_2_1hot[l]] = 1
    return x, np.concatenate((y, label))