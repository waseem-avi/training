class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def build_tree(nodes):
    if not nodes:
        return None

    root = TreeNode(nodes[0])
    queue = [root]
    i = 1

    while queue:
        node = queue.pop(0)

        if i < len(nodes) and nodes[i] is not None:
            node.left = TreeNode(nodes[i])
            queue.append(node.left)
        i += 1

        if i < len(nodes) and nodes[i] is not None:
            node.right = TreeNode(nodes[i])
            queue.append(node.right)
        i += 1

    return root

def burn_time(root, start_node):
    if not root:
        return 0

    def dfs(node, target, depth):
        if not node:
            return 0

        if node.val == target:
            return depth

        left_depth = dfs(node.left, target, depth + 1)
        if left_depth:
            return left_depth

        right_depth = dfs(node.right, target, depth + 1)
        if right_depth:
            return right_depth

    max_depth = dfs(root, start_node, 0)
    return max_depth

# Input
nodes = list(map(int, input().split()))
start_node = int(input())

# Build the tree
root = build_tree(nodes)

# Output
print(burn_time(root, start_node))
