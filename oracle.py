from os import listdir
import codecs


class Oracle:
    def __init__(self, data_dir='data/', langs=None):
        if langs is None:
            langs = ['fr']
        self._oracle = {}
        self.data_dir = data_dir
        self.langs = langs
        for lang in self.langs:
            self._load(lang)

    def _load(self, lang):
        lang_dir = self.data_dir + lang
        for filename in listdir(lang_dir):
            with codecs.open(lang_dir + '/' + filename, encoding='utf-8') as file:
                sentence = [('__ROOT__', None, None)]
                # line_nb = 0  # DEBUG
                for line in file:
                    # line_nb += 1  # DEBUG
                    if line.startswith('#') or len(line) == 1:
                        if len(sentence) == 1:  # ignore comments and empty lines
                            continue

                        # new sentence
                        for word, parent, link in sentence[1:]:
                            if word not in self._oracle:
                                self._oracle[word] = {}
                            self._oracle[word][sentence[parent][0]] = link
                        sentence = [('__ROOT__', None, None)]
                        continue

                    # ASSUMPTION : words are always in the sentence order
                    line = line.split('\t')
                    word = line[1]

                    if line[6] == '_':
                        continue

                    parent = int(line[6])
                    link = line[7]

                    sentence.append((word, parent, link))

    def get(self, word1, word2):
        if word1 not in self._oracle or word2 not in self._oracle[word1]:
            if word2 not in self._oracle or word1 not in self._oracle[word2]:
                return None
            return self._oracle[word2][word1]
        return self._oracle[word1][word2]


if __name__ == '__main__':
    o = Oracle()
    print('ok')
