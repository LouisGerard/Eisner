import utils
import numpy as np

def create_example(w1, w2, positive=True):
    dist = int(w2[index_i]) - int(w1[index_i])

    pos1 = np.zeros(len(utils.pos_2_1hot))
    pos1[utils.pos_2_1hot[w1[pos_i]]] = 1

    pos2 = np.zeros(len(utils.pos_2_1hot))
    pos2[utils.pos_2_1hot[w2[pos_i]]] = 1
    
    x = np.concatenate(([dist], pos1, pos2))
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

index_i = 0
pos_i = 1
governor_i = 2
label_i = 3

sentences_fr_train = utils.read_conllu("UD_French-GSD/fr_gsd-ud-train.conllu")
sentences_fr_dev = utils.read_conllu("UD_French-GSD/fr_gsd-ud-dev.conllu")
sentences_fr_test = utils.read_conllu("UD_French-GSD/fr_gsd-ud-test.conllu")

sentences_en_train = utils.read_conllu("UD_English-LinES/en_lines-ud-train.conllu")
sentences_en_dev = utils.read_conllu("UD_English-LinES/en_lines-ud-dev.conllu")
sentences_en_test = utils.read_conllu("UD_English-LinES/en_lines-ud-test.conllu")

sentences_nl_train = utils.read_conllu("UD_Dutch-LassySmall/nl_lassysmall-ud-train.conllu")
sentences_nl_dev = utils.read_conllu("UD_Dutch-LassySmall/nl_lassysmall-ud-dev.conllu")
sentences_nl_test = utils.read_conllu("UD_Dutch-LassySmall/nl_lassysmall-ud-test.conllu")

sentences_ja_train = utils.read_conllu("UD_Japanese-GSD-master/ja_gsd-ud-train.conllu")
sentences_ja_dev = utils.read_conllu("UD_Japanese-GSD-master/ja_gsd-ud-dev.conllu")
sentences_ja_test = utils.read_conllu("UD_Japanese-GSD-master/ja_gsd-ud-test.conllu")