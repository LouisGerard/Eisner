class Word:
    def __init__(self, word):
        self.word = word
        self.parent = None
        self.childs = []

    def adopt(self, child, _callback=True):
        self.childs.append(child)
        if _callback:
            child.adopted(self, _callback=False)

    def adopted(self, parent, _callback=True):
        self.parent = parent
        if _callback:
            parent.adopt(self, _callback=False)
