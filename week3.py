# Insert interval
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []
        
        for i in range(len(intervals)):
            if newInterval[1] < intervals[i][0]: #ending of new interval is less than starting of one of the intervals (not overlapping)
                res.append(newInterval)
                return res + intervals[i:]
            elif newInterval[0] > intervals[i][1]: #start of new interval is greater than end of one of the intervals (not overlapping)
                res.append(intervals[i])
            else: #when intervals are overlapping, we will merge the intervals
                newInterval = [min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])]
                
        res.append(newInterval)
        return res


# 01 matrix
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        if not mat:
            return mat

        rows, cols = len(mat), len(mat[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        queue = deque()

        # Initialize queue and set initial distances
        for r in range(rows):
            for c in range(cols):
                if mat[r][c] == 0:
                    queue.append((r, c))
                else:
                    mat[r][c] = float('inf')

        # BFS
        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and mat[nr][nc] > mat[r][c] + 1:
                    mat[nr][nc] = mat[r][c] + 1
                    queue.append((nr, nc))

        return mat
        

    
# K Closest Points to Origin
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # Function to calculate squared distance from origin
        def squared_distance(point):
            x, y = point
            return x**2 + y**2

        # Create a min-heap with squared distances and points
        min_heap = [(squared_distance(point), point) for point in points]
        heapq.heapify(min_heap)

        # Extract the k closest points
        closest_points = [heapq.heappop(min_heap)[1] for _ in range(k)]

        return closest_points
    

# Longest Substring Without Repeating Characters
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_set = set()
        left = 0
        max_length = 0

        for right in range(len(s)):
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            char_set.add(s[right])
            max_length = max(max_length, right - left + 1)

        return max_length
    
# 3Sum
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []

        for i in range(len(nums) - 2):
            # Avoid duplicates for the first element
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            left, right = i + 1, len(nums) - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total < 0:
                    left += 1
                elif total > 0:
                    right -= 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    # Skip duplicates for the second and third elements
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1

        return result
    

# Binary Tree Level Order Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            level = []
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level)

        return result
    

# Clone Graph
if not node:
            return None

        # Dictionary to keep track of cloned nodes
        cloned_nodes = {}

        def dfs(node):
            if node in cloned_nodes:
                # Return the clone if it's already created
                return cloned_nodes[node]

            # Create a clone for the current node
            clone = Node(node.val)
            cloned_nodes[node] = clone

            # Recursively clone the neighbors
            for neighbor in node.neighbors:
                clone.neighbors.append(dfs(neighbor))

            return clone

        return dfs(node)


# Evaluate Reverse Polish Notation
stack = []

        for token in tokens:
            if token in "+-*/":
                # Pop the top two elements for the operation
                num2, num1 = stack.pop(), stack.pop()
                
                if token == '+':
                    stack.append(num1 + num2)
                elif token == '-':
                    stack.append(num1 - num2)
                elif token == '*':
                    stack.append(num1 * num2)
                elif token == '/':
                    # Truncate towards zero
                    stack.append(int(num1 / num2))
            else:
                # Push numbers onto the stack
                stack.append(int(token))

        return stack[0]