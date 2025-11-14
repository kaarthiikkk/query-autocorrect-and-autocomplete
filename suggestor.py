import heapq
from trie import Trie

def autocomplete(trie, prefix, k=5):
    node = trie.search(prefix)
    if not node:
        return []

    suggestions = []

    def dfs(current_node, path):
        if current_node.is_end:
            heapq.heappush(suggestions, (-current_node.freq, path))
        for char, child in current_node.children.items():
            dfs(child, path + char)

    dfs(node, prefix)
    top_k = heapq.nsmallest(k, suggestions)
    return [word for _, word in top_k]