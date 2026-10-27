import heapq


def autocomplete(trie, prefix, k=5):
    """
    Returns the top-k autocomplete suggestions
    based on word frequency.
    """

    node = trie.search(prefix)

    if node is None:
        return []

    suggestions = []

    def dfs(current_node, path):

        if current_node.is_end:
            heapq.heappush(
                suggestions,
                (-current_node.freq, path)
            )

        for char, child in current_node.children.items():
            dfs(
                child,
                path + char
            )

    dfs(node, prefix)

    top_k = heapq.nsmallest(
        k,
        suggestions
    )

    return [
        word
        for _, word in top_k
    ]