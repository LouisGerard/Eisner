import numpy as np

import conll17_ud_eval as las

def read_conllu(conllu_filename, features_enabled=[0, 3, 6, 7], root=[0, 'ROOT', 0, 'root']):
    sentences = []

    features = ['INDEX', 'FORM', 'LEMMA', 'POS', 'X1', 'MORPHO', 'GOV', 'LABEL', 'X2', 'X3']

    columns = []
    for i, f in enumerate(features):
        if i in features_enabled:
            columns.append(f)

    with open(conllu_filename, 'r', encoding='utf-8') as conllu_file:
        sentence = []
        for i in range(len(features_enabled)):
            sentence.append([root[i]])
        for line in conllu_file:
            if line[0] == '\n':
                if len(sentence) > 1:
                    sentences.append(np.array(sentence).T)
                    sentence = []
                    for i in range(len(features_enabled)):
                        sentence.append([root[i]])
            elif line[0] != '#':
                tokens = line.split('\t')
                if '-' not in tokens[0]:
                    for i, j in enumerate(features_enabled):
                        sentence[i].append(tokens[j])
    return sentences

def is_projective(sentence, f_module):
    edges = []
    for w1 in sentence:
        if w1[f_module.index_i] == '0':
            continue
        
        i, j = int(w1[f_module.index_i]), int(w1[f_module.governor_i])
        
        if j < i:
            i, j = j, i
        
        for (i2, j2) in edges:
            i_in = False
            j_in = False
            
            if i == i2 or i == j2 or j == i2 or j == j2:
                continue
            
            if i2 <= i <= j2:
                i_in = True
            if i2 <= j <= j2:
                j_in = True
            
            if i_in != j_in:
                return False
            
        edges.append((i, j))
    return True

def count_projectives(sentences, f_module):
    counter_p = 0
    for s in sentences:
        if not is_projective(s, f_module):
            counter_p += 1
    return counter_p, (counter_p / len(sentences) * 100)

pos_2_1hot = {
    'ADJ': 1,
    'ADP': 2,
    'ADV': 3,
    'AUX': 4,
    'CCONJ': 5,
    'DET': 6,
    'INTJ': 7,
    'NOUN': 8,
    'NUM': 9,
    'PART': 10,
    'PRON': 11,
    'PROPN': 12,
    'PUNCT': 13,
    'SCONJ': 14,
    'SYM': 15,
    'VERB': 16,
    'X': 17,
    'ROOT': 0
}

labels_2_1hot = {
    'acl': 0,
    'advcl': 1,
    'advmod': 2,
    'amod': 3,
    'appos': 4,
    'aux': 5,
    'case': 6,
    'cc': 7,
    'ccomp': 8,
    'clf': 9,
    'compound': 10,
    'conj': 11,
    'cop': 12,
    'csubj': 13,
    'dep': 14,
    'det': 15,
    'discourse': 16,
    'dislocated': 17,
    'expl': 18,
    'fixed': 19,
    'flat': 20,
    'goeswith': 21,
    'iobj': 22,
    'list': 23,
    'mark': 24,
    'nmod': 25,
    'nsubj': 26,
    'nummod': 27,
    'obj': 28,
    'obl': 29,
    'orphan': 30,
    'parataxis': 31,
    'punct': 32,
    'reparandum': 33,
    'root': 34,
    'vocative': 35,
    'xcomp': 36
}
onehot_2_label = {v: k for k, v in labels_2_1hot.items()}

def create_dataset(sentences, f_module, *args, with_negatives=False):
    x = []
    y = []
    for s in sentences:
        for w1 in s:
            if w1[f_module.index_i] == '0':
                continue
            w2 = s[int(w1[f_module.governor_i])]
            
            if w1[f_module.index_i] > w2[f_module.index_i]:
                w2, w1 = w1, w2

            x_token, y_token = f_module.create_example(w1, w2, *args)

            x.append(x_token)
            y.append(y_token)
            
            if with_negatives:
                i = np.random.randint(len(s) - 2)
                if i >= int(w1[f_module.index_i]):
                    i += 1
                if i >= int(w2[f_module.index_i]):
                    i += 1
                w2_negative = s[i]
                x_token, y_token = f_module.create_example(w1, w2, *args, positive=False)
                x.append(x_token)
                y.append(y_token)

    return np.array(x), np.array(y)

def eisner(sentence, f_module, *args, model=None, perfect=False):
    n = sentence.shape[0]

    full_left = []
    full_right = []
    part_left = []
    part_right = []
    
    part_max = []
    full_left_max = []
    full_right_max = []
    
    labels = []
    
    for i in range(n):
        full_left.append([0])
        full_right.append([0])
        part_left.append([0])
        part_right.append([0])
        
        part_max.append([0])
        full_left_max.append([0])
        full_right_max.append([0])
        
        labels.append([])

    for m in range(1, n):
        for i1 in range(n - m):
            i2 = i1 + m
            
            if perfect:
                prediction = np.zeros(39)
                prediction[0] = 0
                prediction[1] = 0
                label = None
                if sentence[i1][f_module.governor_i] == sentence[i2][f_module.index_i]:
                    prediction[0] = 1
                    label = sentence[i1][f_module.label_i]
                elif sentence[i2][f_module.governor_i] == sentence[i1][f_module.index_i]:
                    prediction[1] = 1
                    label = sentence[i2][f_module.label_i]
            else:
                x, _ = f_module.create_example(sentence[i1], sentence[i2], *args)
                if type(x) != list:
                    x = x.reshape(1, -1)
                prediction = model.predict(x)[0]

                i = np.argmax(prediction[2:])
                label = onehot_2_label[i]
                
            labels[i1].append(label)

            max_full = -1
            max_q = -1
            for q in range(i2 - i1):
                q_line = q + i1
                current = full_left[i1][q] + full_right[q_line + 1][i2 - q_line - 1]
                
                if current > max_full:
                    max_full = current
                    max_q = q_line
            
            part_left[i1].append(max_full + prediction[1])
            part_right[i1].append(max_full + prediction[0])
            part_max[i1].append(max_q)
            
            max_full_l = -1
            max_full_r = -1
            max_q_l = -1
            max_q_r = -1
            for q in range(i2 - i1):
                q_line = q + i1
                current_r = full_right[i1][q] + part_right[q_line][i2 - q_line]
                current_l = part_left[i1][q + 1] + full_left[q_line + 1][i2 - q_line - 1]

                if current_l > max_full_l:
                    max_full_l = current_l
                    max_q_l = q_line + 1
                if current_r > max_full_r:
                    max_full_r = current_r
                    max_q_r = q_line

            full_left[i1].append(max_full_l)
            full_right[i1].append(max_full_r)
            full_left_max[i1].append(max_q_l)
            full_right_max[i1].append(max_q_r)
                        
    return full_left_max, full_right_max, part_max, labels

def color(val):
    if val:
        color = 'green'
    else:
        color = 'red'
    return 'background-color: %s' % color

def compare(tab1, tab2):
    df = (tab1 == tab2) | tab2.isnull()
    return df.style.applymap(color)

def decompose_full(sentence, f_module, i1, i2, full_left, full_right, part, labels, left=True):
#     print('full', left, i1, i2)
    if i1 < i2:
        if left:
            q = full_left[i1][i2 - i1]
            f1 = decompose_part
            f2 = decompose_full
        else:
            q = full_right[i1][i2 - i1]
            f1 = decompose_full
            f2 = decompose_part
        sentence = f1(sentence, f_module, i1, q, full_left, full_right, part, labels, left)
        sentence = f2(sentence, f_module, q, i2, full_left, full_right, part, labels, left)
    return sentence

def decompose_part(sentence, f_module, i1, i2, full_left, full_right, part, labels, left=True):
#     print('part', left, i1, i2)
    if left:
        sentence[i2][f_module.governor_i] = i1
        sentence[i2][f_module.label_i] = labels[i1][i2 - i1 - 1]
    else:
        sentence[i1][f_module.governor_i] = i2
        sentence[i1][f_module.label_i] = labels[i1][i2 - i1 - 1]
        
    if i1 < i2:
        q = part[i1][i2 - i1]
        sentence = decompose_full(sentence, f_module, i1, q, full_left, full_right, part, labels, True)
        sentence = decompose_full(sentence, f_module, q + 1, i2, full_left, full_right, part, labels, False)
    return sentence
        
def predict_sentence(sentence, f_module, full_left, full_right, part, labels):
    sentence_predicted = sentence.copy()
    for i in range(len(part[0])):
        part[0][i] = 0
    decompose_full(sentence_predicted, f_module, 0, len(sentence) - 1, full_left, full_right, part, labels)
    return sentence_predicted

def predict_sentences(filename, f_module, *args, model=None, features_enabled=[0, 3, 6, 7], root=[0, 'ROOT', 0, 'root'], perfect=False, sentences_callback=None):
    sentences_test = read_conllu(filename, features_enabled, root)

    if sentences_callback is not None:
        sentences_test = sentences_callback(sentences_test)

    for s in range(len(sentences_test)):
        full_left, full_right, part, labels = eisner(sentences_test[s], f_module, *args, model=model, perfect=perfect)
        sentences_test[s] = predict_sentence(sentences_test[s], f_module, full_left, full_right, part, labels)
    
    return sentences_test

def write_conllu(sentences, f_module, filename_in, filename_out):
#     features = ['INDEX', 'FORM', 'LEMMA', 'POS', 'X1', 'MORPHO', 'GOV', 'LABEL', 'X2', 'X3']
    with open(filename_in, 'r', encoding='utf-8') as fin, open(filename_out, 'w', encoding='utf-8') as fout:
        sentence_i = 0
        sentence_begin = True
        for line in fin:
            if line[0] == '\n':
                fout.write(line)
                if not sentence_begin:
                    sentence_i += 1
                    sentence_begin = True
            elif line[0] != '#':
                sentence_begin = False
                tokens = line.split('\t')
                if '-' in tokens[0]:
                    fout.write(line)
                    continue
                if int(tokens[0]) - 1 >= len(sentences[sentence_i]):
                    print(line)
                    print(sentences[sentence_i])
                    print(sentence_i)
                    break
                tokens[6] = sentences[sentence_i][int(tokens[0])][f_module.governor_i]
                tokens[7] = sentences[sentence_i][int(tokens[0])][f_module.label_i]
                fout.write('\t'.join(tokens))

def score_las(filename_test, filename_gold):
    score = 0
    with open(filename_test, 'r') as ftest, \
            open(filename_gold, 'r') as fgold:
        test = las.load_conllu(ftest)
        gold = las.load_conllu(fgold)
        score = las.evaluate(gold, test)['LAS'].f1
    return score

