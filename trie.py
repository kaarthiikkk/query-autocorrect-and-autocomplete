class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.freq = 0


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, frequency=1):
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()

            node = node.children[char]

        node.is_end = True
        node.freq = frequency

    def search(self, prefix):
        node = self.root

        for char in prefix:
            if char not in node.children:
                return None

            node = node.children[char]

        return node

    def collect_words(self, node=None, prefix="", words=None):
        if words is None:
            words = []

        if node is None:
            node = self.root

        if node.is_end:
            words.append((prefix, node.freq))

        for char, child in node.children.items():
            self.collect_words(
                child,
                prefix + char,
                words
            )

        return words