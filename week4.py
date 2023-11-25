# Course Schedule (hard)
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # For every course, note the prerequisites
        graph = {i: [] for i in range(numCourses)}
        for a, b in prerequisites:
            graph[a].append(b)

        # A function to perform DFS on the graph
        def dfs(course, visited, stack):
            if course in stack:
                # Cycle detected
                return False
            if course in visited:
                # Already visited this node and no cycle was detected
                return True
            
            # Mark the course as visited and add to the current stack
            visited.add(course)
            stack.add(course)

            # Perform DFS on the prerequisites of the current course
            for prereq in graph[course]:
                if not dfs(prereq, visited, stack):
                    return False
            
            # Remove the course from the stack after exploring its prerequisites
            stack.remove(course)
            return True

        

        # Check for each course if you can finish it
        for course in range(numCourses):
            if not dfs(course, set(), set()):
                return False

        return True



# Implement Trie (Prefix Tree)

# useful video: https://www.youtube.com/watch?v=zIjfhVPRZCg
# In this Trie implementation, each node can have more than two children. In fact, the number of children for each node in a Trie is not fixed and generally depends on the character set of the strings being stored. For instance, if the Trie is designed to store English words, each node can have up to 26 children
class TrieNode:
    # Helper class representing each node in the Trie
    def __init__(self):
        self.children = {}
        self.isEndOfWord = False

class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.isEndOfWord = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.isEndOfWord

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

        


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

# Coin change
# https://www.youtube.com/watch?v=H9bfqozjoqs
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # Initialize a list to store the minimum coins needed for each amount
        # Set initial value to a large number, greater than the maximum possible number of coins
        dp = [float('inf')] * (amount + 1)
        
        # Base case: 0 coins are needed to make amount 0
        dp[0] = 0

        # Iterate over all the amounts from 1 to 'amount'
        for a in range(1, amount + 1):
            # Iterate over each coin
            for coin in coins:
                if a - coin >= 0:
                    # Update the dp array if using the current coin results in a smaller number of coins
                    dp[a] = min(dp[a], 1 + dp[a - coin])

        # If dp[amount] is still infinity, it means the amount cannot be made up by any combination of the coins
        return -1 if dp[amount] == float('inf') else dp[amount]
    

# Product of Array Except Self
# Left Products: For each element i, compute the product of all elements before i.
# Right Products: Similarly, compute the product of all elements after i.
# Result: For each element i, the result is the product of its left and right products.
# https://www.youtube.com/watch?v=bNvIQI2wAjk

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        length = len(nums)
        answer = [1] * length

        # Calculate left products
        left_product = 1
        for i in range(1, length):
            left_product *= nums[i - 1]
            answer[i] *= left_product

        # Calculate right products in reverse
        right_product = 1
        for i in range(length - 2, -1, -1):
            right_product *= nums[i + 1]
            answer[i] *= right_product

        return answer
    


# Min stack

# Key is to keep track of minimum value for each addition to stack (when that item is removed, revert to previous min value)
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        # If the min stack is empty or the current value is less than the current minimum
        # push the current value onto the min stack
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        # Pop from both the main stack and the min stack (if the popped values are the same)
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        # Return the top element of the main stack
        return self.stack[-1]

    def getMin(self) -> int:
        # Return the top element of the min stack (which is the current minimum)
        return self.min_stack[-1]
        

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


# Validate Binary Search Tree

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def validate(node, low=-float('inf'), high=float('inf')):
            # An empty tree is a valid BST
            if not node:
                return True

            # The current node's value must be between low and high
            if not (low < node.val < high):
                return False

            # Recursively check the left and right subtree
            return (validate(node.left, low, node.val) and
                    validate(node.right, node.val, high))

        return validate(root)
    

# Number of Islands (recursive dfs). This approach modifies the input grid to mark visited cells (0s)
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0

        rows, cols = len(grid), len(grid[0])
        num_islands = 0

        def dfs(r, c):
            if r < 0 or c < 0 or r >= rows or c >= cols or grid[r][c] == '0':
                return
            grid[r][c] = '0'  # Mark as visited
            dfs(r + 1, c)  # Down
            dfs(r - 1, c)  # Up
            dfs(r, c + 1)  # Right
            dfs(r, c - 1)  # Left

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    dfs(r, c)
                    num_islands += 1  # Increase count for each starting point of an island

        return num_islands


#BFS version
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0

        rows, cols = len(grid), len(grid[0])
        visit = set()
        num_islands = 0

        def bfs(r, c):
            queue = collections.deque()
            visit.add((r,c))
            queue.append((r, c))
            while queue:
                row, col = queue.popleft() # change popleft to pop to do DFS instead (iterative)
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    if (r in range(rows) and
                        c in range(cols) and
                        grid[r][c] == '1' and
                        (r, c) not in visit):
                        visit.add((r, c))
                        queue.append((r, c))

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1' and (r,c) not in visit:
                    bfs(r, c)
                    num_islands += 1  # Increase count for each starting point of an island

        return num_islands

# Rotting oranges (use BFS but remove all rotten oranges at once)
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        fresh_count = 0
        queue = deque()

        # Initialize the queue with all rotten oranges and count fresh oranges
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c))
                elif grid[r][c] == 1:
                    fresh_count += 1

        # Directions for adjacent cells (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        minutes = 0

        # BFS
        while queue and fresh_count > 0:
            minutes += 1
            for _ in range(len(queue)):
                x, y = queue.popleft()
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1:
                        grid[nx][ny] = 2
                        fresh_count -= 1
                        queue.append((nx, ny))

        return minutes if fresh_count == 0 else -1
