# Maximum number of characters in the ASCII character set
MAX_CHARACTERS = 256

# Maximum length of an input word
MAX_WORD_LEN = 500

# Trie Node
class TrieNode:
    def __init__(self):
        # Child nodes representing each character
        self.child = [None] * MAX_CHARACTERS
        # To store frequency
        self.freq = 1

# Function to create a new trie node
def newTrieNode():
    newNode = TrieNode()
    return newNode

# Method to insert a new string into Trie
def insert(root, word):
    # Length of the word
    len_word = len(word)
    pCrawl = root

    # Traverse over the length of the given word
    for level in range(len_word):
        # Get the index of the child node from the ASCII value of the current character
        index = ord(word[level])

        # Create a new child if it does not exist already
        if not pCrawl.child[index]:
            pCrawl.child[index] = newTrieNode()
        else:
            pCrawl.child[index].freq += 1

        # Move to the child
        pCrawl = pCrawl.child[index]

# This function prints unique prefixes for every word stored in the Trie.
# Prefixes, one by one, are stored in the prefix[] list.
# 'ind' is the current index of the prefix[]
def findPrefixesUtil(root, prefix, ind):
    # Corner case
    if not root:
        return

    # Base case
    if root.freq == 1:
        prefix[ind] = ""
        print("".join(prefix[:ind]), end=" ")
        return

    for i in range(MAX_CHARACTERS):
        if root.child[i]:
            prefix[ind] = chr(i)
            findPrefixesUtil(root.child[i], prefix, ind + 1)

# Function to print all prefixes that uniquely represent all words in arr[0..n-1]
def findPrefixes(words, n):
    # Construct a Trie for all words
    root = newTrieNode()
    root.freq = 0

    for word in words:
        insert(root, word)

    # Create an array to store all prefixes
    prefix = [None] * MAX_WORD_LEN

    # Print all prefixes using Trie Traversal
    findPrefixesUtil(root, prefix, 0)

# Driver function
if __name__ == "__main__":
    words = ["zebra", "dog", "duck", "dove"]
    num_words = len(words)
    findPrefixes(words, num_words)
