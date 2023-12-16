# container with most water
class Solution:
    def maxArea(self, height: List[int]) -> int:
        # initialize left pointer at left and right pointer all the way on the right
        # calculate max pointer, then shift the pointer that has smaller height so that it can potentially increase, while constantly track

        res = 0
        
        l, r = 0, len(height) - 1

        while l < r: 
            area = (r-l) * min(height[l],height[r])
            res = max(res, area)

            if height[l] < height[r]:
                l += 1
            # if other condition (or if hit a place where left and right height are equal, in which case it does not matter which one your shift)
            else:
                r -= 1

        return res
    

# Letter combinations of a phone number

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        res = []

        digitToChar = {
            '2':'abc',
            '3':'def',
            '4':'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'qprs',
            '8': 'tuv',
            '9': 'wxyz'
        }

        def backtrack(i, curStr):
            if len(curStr) == len(digits):
                res.append(curStr)
                return
            
            for c in digitToChar[digits[i]]:
                backtrack(i+1, curStr+c)
        
        if digits:
            backtrack(0, "")
        
        return res


# Word Search
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        ROWS, COLS = len(board), len(board[0])
        path = set()

        def dfs(r, c, i):
            if i == len(word):
                return True
            
            if (r < 0 or c < 0 or r >= ROWS or c >= COLS or word[i] != board[r][c] or (r,c) in path):
                return False

            path.add((r,c))
            res = (dfs(r+1, c, i +1) or dfs(r-1, c, i + 1) or dfs(r, c + 1, i +1) or dfs(r, c-1, i +1))

            # backtracking so that other nodes can be visited
            path.remove((r,c))
            return res
        
        for r in range(ROWS):
            for c in range(COLS):
                if dfs(r,c,0): return True
        
        return False
    
# Ah, I see where the confusion lies. You are correct that the DFS algorithm tests all four directions in separate recursive calls, and each call independently explores a distinct path. The need for backtracking (and thus `path.remove((r, c))`) arises not because those paths won't be tested elsewhere, but because of how recursion and the path set (`path`) are being handled in this specific implementation.

#Here's a more detailed explanation to clarify:

#1. **Independent Recursive Calls**: When the DFS algorithm is at a particular cell, it does indeed make four recursive calls (if possible), each exploring a different direction. Each of these calls is independent and will explore its path to completion, whether it leads to a dead end or finds a part of the word.

#2. **The Role of the Path Set**: The `path` set is shared across all these recursive calls. It tracks all the cells that are part of the current recursion path. This is important because a cell used in one part of the word cannot be reused later in the same path.

#3. **Why Backtracking is Needed**: Let's say one recursive call goes down a path and ends up at a dead end (i.e., it cannot complete the word). At this point, the cells that were part of this unsuccessful path need to be available for other recursive calls that are exploring different paths. 

#4. **Clearing the Path with `path.remove`**: When a recursive call returns (either because it found a part of the word or hit a dead end), the cells it visited are no longer part of an active path. They need to be removed from the `path` set so that other recursive calls can use them. If we don't remove them from `path`, these cells would be incorrectly marked as part of the ongoing recursion path, even though they're actually part of a path that has already been abandoned.

#5. **Example Scenario**: Imagine you are searching for "CAT" and the algorithm follows the path C -> A -> T, but the next required letter isn't adjacent. This path fails, so the algorithm backtracks. Without removing 'T' and then 'A' from `path`, if there's another recursive call that needs to use these cells for a different path (like A -> T in a different part of the board), it won't be able to, incorrectly thinking they're still part of an ongoing search.

#In summary, while it's true that each direction from a cell is tested in a different recursive call, `path.remove((r, c))` is necessary to ensure that once a recursive call completes (successfully or not), the cells it used are marked as no longer part of the current path. This allows other recursive calls to use these cells for their independent path explorations. Without this step, the shared `path` set would inaccurately reflect the cells in play for the current recursion path.


# Even though the recursive calls in the depth-first search (DFS) might appear to be happening in parallel due to their nested nature, they are actually executed sequentially. The path variable is shared among these calls, and its state is critical for ensuring that each recursive call correctly tracks the cells it has visited. Let's clarify this with a focus on how recursion and the shared path variable work in this context:

# Sequential Execution of Recursive Calls:

# When DFS is called for a cell, it explores each of the four directions (up, down, left, right) one after the other, not simultaneously.
# Each call to dfs(r, c, i) completes its execution before the next call starts. This means that the algorithm fully explores one direction (and all its subsequent paths) before moving on to the next direction.


# Find all anaagrams in a string

# sliding window (explained well here: https://www.youtube.com/watch?v=G8xtZy0fDKg)

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(p) > len(s): return []

        pCount, sCount = {}, {}

        # not using from collections import Counter because maybe not allowed
        # initalize the sliding window
        for i in range(len(p)):
            pCount[p[i]] = 1 + pCount.get(p[i],0)
            sCount[s[i]] = 1 + sCount.get(s[i], 0)
        
        res = [0] if sCount == pCount else []
        l = 0
        # move right pointer
        for r in range(len(p), len(s)):
            sCount[s[r]] = 1 + sCount.get(s[r], 0)
            sCount[s[l]] -= 1

            if sCount[s[l]] == 0:
                sCount.pop(s[l])

            l += 1

            if sCount == pCount:
                res.append(l)
        
        return res
    

# Minimum height tree
from typing import List
from collections import defaultdict

class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        # Base cases
        if n <= 2:
            return [i for i in range(n)]

        # Create an adjacency list
        neighbors = defaultdict(set)
        for start, end in edges:
            neighbors[start].add(end)
            neighbors[end].add(start)

        # Initialize the first layer of leaves
        leaves = [i for i in range(n) if len(neighbors[i]) == 1]

        # Trim the leaves until reaching the centroids
        remaining_nodes = n
        while remaining_nodes > 2:
            remaining_nodes -= len(leaves)
            new_leaves = []
            for leaf in leaves:
                # The only neighbor left for the leaf
                neighbor = neighbors[leaf].pop()
                # Remove the leaf from its neighbor's set
                neighbors[neighbor].remove(leaf)
                if len(neighbors[neighbor]) == 1:
                    new_leaves.append(neighbor)
            leaves = new_leaves

        return leaves



# ### Code Breakdown

# #### 1. Define the `Solution` class and `findMinHeightTrees` method:
# ```python
# class Solution:
#     def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
# ```
# - We're defining a class named `Solution`.
# - Inside `Solution`, there's a method `findMinHeightTrees` that takes two arguments:
#   - `n`: The total number of nodes in the tree.
#   - `edges`: A list of edges, where each edge is a pair of nodes indicating a connection between them.

# #### 2. Handle Base Cases:
# ```python
# if n <= 2:
#     return [i for i in range(n)]
# ```
# - If the tree has 2 or fewer nodes, then each node is a valid root for an MHT (because the tree is already as short as it can be).
# - We return a list of all nodes.

# #### 3. Create an Adjacency List:
# ```python
# neighbors = defaultdict(set)
# for start, end in edges:
#     neighbors[start].add(end)
#     neighbors[end].add(start)
# ```
# - We're creating a map (`neighbors`) to keep track of how nodes are connected.
# - For each connection (`edge`), we add each node to the set of neighbors of the other node.
# - This helps us quickly find all nodes connected to any given node.

# #### 4. Initialize the First Layer of Leaves:
# ```python
# leaves = [i for i in range(n) if len(neighbors[i]) == 1]
# ```
# - We identify the "leaves" of the tree.
# - A leaf is a node with only one connection (neighbor).
# - We make a list of all such nodes.

# #### 5. Trim the Leaves Iteratively:
# ```python
# remaining_nodes = n
# while remaining_nodes > 2:
#     remaining_nodes -= len(leaves)
#     new_leaves = []
#     for leaf in leaves:
#         ...
# ```
# - We keep trimming the leaves until the number of remaining nodes is 2 or less.
# - Each time we trim a leaf, we reduce the count of `remaining_nodes`.
# - We prepare to collect the next set of leaves in `new_leaves`.

# #### 6. Process Each Leaf:
# ```python
# neighbor = neighbors[leaf].pop()
# neighbors[neighbor].remove(leaf)
# if len(neighbors[neighbor]) == 1:
#     new_leaves.append(neighbor)
# ```
# - For each leaf, we find its only neighbor and remove the leaf from the neighbor's set of connections.
# - If this action turns the neighbor into a leaf (only one connection remaining), we add it to `new_leaves`.

# #### 7. Update the Leaves List:
# ```python
# leaves = new_leaves
# ```
# - We update our list of leaves to the new leaves found in the current iteration.

# #### 8. Return the Remaining Nodes:
# ```python
# return leaves
# ```
# - After trimming all possible leaves, the nodes remaining in the `leaves` list are the best candidates for being the roots of MHTs.
# - We return this list as the final result.


# Handle Simple Cases:

# If there are 1 or 2 points, they are automatically the roots of MHTs.
# Build a Map of Connections:

# We create a map showing how each point is connected to others.
# Find the Outer Points (Leaves):

# Leaves are points with only one connection. They are like the tips of the branches.
# Trim the Leaves:

# We repeatedly remove these outer points. When a leaf is removed, the point it was connected to might become a new leaf.
# This is like pruning the branches from the tips inward.
# Stop When We Reach the Center:

# We stop when we can't trim anymore without making the tree disconnected.
# The points left are the best candidates for being the roots of MHTs.
# Result:

# The remaining points are the roots that make the tree as short as possible.
# https://www.youtube.com/watch?v=ivl6BHJVcB0

# Task scheduler

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        # good explanation of problem
        # https://www.youtube.com/watch?v=s8p8ukTyA2I
        # start with the most frequently repeating character so can use up idle time
        # use max heap to determine what is max at all times with log(n) time
        # no native max heap, so use min heap and take negative of values before adding

        count = Counter(tasks)
        # want negative of value before you heapify it
        # important thing here is that you don't need to track actual values (you just have to know that they are different and how many there are!!!)
        maxHeap = [-cnt for cnt in count.values()]
        # makes it a min heap (but maxheap with negative)
        heapq.heapify(maxHeap)

        time = 0
        q = deque() # pairs of [-cnt, idleTime]

        while maxHeap or q:
            time += 1

            if maxHeap:
                # decrement the count (add because using negative values)
                cnt = 1 + heapq.heappop(maxHeap)
                # if cnt is non-zero, append to queue
                if cnt:
                    q.append([cnt, time + n])
            
            if q and q[0][1] == time:
                heapq.heappush(maxHeap, q.popleft()[0])
        return time

# LRU cache
class Node:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    # more of a design problem than an algorithm problem
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {} # map key to node

        # left = LRU, right = most recent
        self.left, self.right = Node(0,0), Node(0,0)
        self.left.next, self.right.prev = self.right, self.left

    # remove from the list
    def remove(self, node):
        prev, nxt = node.prev, node.next
        prev.next, nxt.prev = nxt, prev
    
    # insert node at right
    def insert(self, node):
        prev, nxt = self.right.prev, self.right
        prev.next = nxt.prev = node
        node.prev, node.next = prev, nxt

    def get(self, key: int) -> int:
        if key in self.cache:
            # remove from list, then insert at right most position when get node
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val

        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        self.cache[key] = Node(key,value)    
        self.insert(self.cache[key])

        if len(self.cache) > self.cap:
            # remove from the list and delete the LRU frm the hashmap
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

