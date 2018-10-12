from os import listdir
import codecs

oracle = {}

data_dir = 'data/'
lang_dir = data_dir + 'fr/'
for filename in listdir(lang_dir):
    with codecs.open(lang_dir + filename, encoding='utf-8') as file:
        sentence = [('__ROOT__', None, None)]
        ignore = False
        line_nb = 0 ################# delete me
        for line in file:
            line_nb += 1 ################## delete me
            if line.startswith('#') or len(line) == 1 or ignore:
                if len(sentence) == 1:  # ignore comments and empty lines
                    ignore = False
                    continue

                if not ignore:
                    # new sentence
                    for word, parent, link in sentence[1:]:
                        if word not in oracle:
                            oracle[word] = {}
                        oracle[word][sentence[parent][0]] = link
                    sentence = [('__ROOT__', None, None)]
                continue

            # ASSUMPTION : words are always in the sentence order
            line = line.split('\t')

            if line[6] == '_':  # no parent ? f*ck you then
                ignore = True
                continue

            word = line[1]
            parent = int(line[6])
            link = line[7]

            sentence.append((word, parent, link))

print('test')
