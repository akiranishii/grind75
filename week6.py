# word break
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # https://www.youtube.com/watch?v=Sx9NNgInc3A
        # 1D array as cache 
        dp = [False] * (len(s) + 1)
        dp[len(s)] = True
        # start, stop (up to but including this value), step
        for i in range(len(s) - 1, -1, -1):
            for w in wordDict:
                if (i + len(w)) <= len(s) and s[i: i +len(w)] == w:
                    dp[i] = dp[i+len(w)]
                if dp[i]:
                    break

        return dp[0]



# Partition Equal Subset Sum
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        # important observation is that the two subsets must equal to half of the total sum

        if sum(nums) % 2:
            return False

        dp = set()
        dp.add(0)
        target = sum(nums) // 2

        for i in range(len(nums) - 1, -1, -1):
            nextDP = set()
            for t in dp:
                nextDP.add(t + nums[i])
                nextDP.add(t)
            dp = nextDP

        return True if target in dp else False
    
# String to integer (atoi)
class Solution:
    def myAtoi(self, s: str) -> int:
        # Define the bounds of a 32-bit signed integer
        INT_MAX = 2**31 - 1
        INT_MIN = -2**31

        # Initialize the index and sign
        i = 0
        sign = 1

        # Remove leading whitespaces
        s = s.strip()

        # Check if the string is empty
        if not s:
            return 0

        # Check if the first character is '-' or '+'
        if s[0] == '-':
            sign = -1
            i += 1
        elif s[0] == '+':
            i += 1

        # Initialize result
        result = 0

        # Iterate over the characters of the string
        while i < len(s) and s[i].isdigit():
            digit = int(s[i])

            # Check for overflow and underflow conditions (before the actual addition is made)
            if result > INT_MAX // 10 or (result == INT_MAX // 10 and digit > INT_MAX % 10):
                return INT_MAX if sign == 1 else INT_MIN

            result = result * 10 + digit
            i += 1

        return sign * result
    

# Spiral matrix
    result = []
        if not matrix:
            return result

        left, right, top, bottom = 0, len(matrix[0]), 0, len(matrix)

        while left < right and top < bottom:
            # Traverse from left to right
            for i in range(left, right):
                result.append(matrix[top][i])
            top += 1

            # Traverse downwards
            for i in range(top, bottom):
                result.append(matrix[i][right - 1])
            right -= 1

            if not (left < right and top < bottom):
                break

            # Traverse from right to left
            for i in range(right - 1, left - 1, -1):
                result.append(matrix[bottom - 1][i])
            bottom -= 1

            # Traverse upwards
            for i in range(bottom - 1, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

        return result


# Subsets
class Solution:
   def subsets(self, nums):
       result = [[]]
       for num in nums:
           result += [curr + [num] for curr in result]
       return result


# or there is also a dfs solution

class Solution:
    def subsets(self, nums):
        res = []
        subset = []
        def dfs(i):
            if i >= len(nums):
               res.append(subset.copy())
               return

            # decision to include nums[i]
            subset.append(nums[i])
            dfs(i + 1)

            # decision to NOT include nums[i]
            subset.pop()
            dfs(i + 1)
        
        dfs(0)
        return res
    

# Binary tree right side view
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        # for each "level" of the tree, we want to the right most node
        # BFS
        if not root:
            return []

        right_view = []
        queue = [root]
        
        while queue:
            level_length = len(queue)
            for i in range(level_length):
                node = queue.pop(0)
                
                if i == level_length - 1:
                    right_view.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return right_view

# could also use deque and popleft

# Longest Palindromic Substring

class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) < 2:
            return s
        
        start = 0
        end = 0
        
        for i in range(len(s)):
            # odd length palindrome
            len1 = self.expandAroundCenter(s, i, i)
            # even length palindrome
            len2 = self.expandAroundCenter(s, i, i + 1)
            length = max(len1, len2)
            
            # if new length is greater than previous length, update start and end (of the longest palindrome)
            if length > end - start:
                start = i - (length - 1) // 2
                end = i + length // 2
        
        return s[start:end+1]
    
    def expandAroundCenter(self, s: str, left: int, right: int) -> int:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        
        return right - left - 1


# Unique paths
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        row = [1] * n
        
        for i in range(m-1):
            newRow = [1] * n
            for j in range(n-2, -1, -1):
                newRow[j] = newRow[j+1] + row[j]
            row = newRow
        return row[0]



    

