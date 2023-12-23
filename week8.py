# Kth Smallest Element in a BST

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        # stop when n == k
        n = 0
        # use stack to itertively convert binary tree to sorted list (https://www.youtube.com/watch?v=5LUXSvjmGCw)
        stack = []
        # keep track of node that you are currently visiting
        cur = root

        # while cur is not null and stack is not empty
        while cur or stack:
            # while cur is not null, visit left (want most left element)
            while cur:
                stack.append(cur)
                cur = cur.left
            
            cur = stack.pop() # most recently added element
            n += 1
            # this will always execute because guarenteed to have k nodes
            if n == k:
                return cur.val

            cur = cur.right


            # there is another recursive solution! 


# Minimum window substring

class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        # https://www.youtube.com/watch?v=jSto0O4AJbM
        
        # edge case
        if t == "": return ""
        
        # initialize dictionaries for two strings
        countT, window = {}, {}
        for c in t:
            countT[c] = 1 + countT.get(c, 0)

        have, need = 0, len(countT)

        # initialize result index store and result length
        res, resLen = [-1,-1], float('infinity')
        # l is the left pointer
        l = 0 

        # r is right pointer
        for r in range(len(s)):
            c = s[r]
            window[c] = 1 + window.get(c,0)

            if c in countT and window[c] == countT[c]:
                have += 1

            while have == need:
                # update our result
                if (r-l+1) < resLen:
                    res = [l, r]
                    resLen = (r - l + 1)
                
                # pop from the left of our window
                window[s[l]] -= 1

                if s[l] in countT and window[s[l]] < countT[s[l]]:
                    have -= 1

                l += 1

        l, r = res
        return s[l:r+1] if resLen != float("infinity") else ""


# Serialize and deserialize binary tree
    
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# https://www.youtube.com/watch?v=u4JAi2JJhI8

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """

        res = []
        
        def dfs(node):
            if not node:
                res.append("N")
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ",".join(res)
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        vals = data.split(",")
        # pointer needs to be a global variable (so make it self)
        self.i = 0

        def dfs():
            if vals[self.i] == "N":
                self.i += 1
                return None
            node = TreeNode(int(vals[self.i]))
            self.i += 1
            node.left = dfs()
            node.right = dfs()
            return node
        
        return dfs()


# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))


# Trapping Rain Water
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
    # https://www.youtube.com/watch?v=ZI2z5pq0TqA
    # for each index, amount of water that can be held is min(maxLeft, maxRight) - height @ index i (note if calculated value is negative, it is zero!)
    # https://www.youtube.com/watch?v=ZI2z5pq0TqA
        if not height:
            return 0

        # initialize index and beginning and end
        l, r = 0, len(height)-1
        leftMax, rightMax = height[l], height[r]
        res = 0

        while l < r:
            # move left pointer
            if leftMax < rightMax:
                l += 1
                leftMax = max(leftMax, height[l])
                res += leftMax - height[l]
            # move right pointer
            else:
                r -= 1
                rightMax = max(rightMax, height[r])
                res += rightMax - height[r]

        return res

# > leftMax = max(leftMax, height[l])

# If leftMax is less than (or equal to) height[l], we set leftMax equal to height[l] and then subtract height[l] from it, which would equal zero.
# On the other hand, if leftMax is greater than height[l], we keep leftMax as is and subtract height[l] from it, which would be a positive value.

# In either case, the result will always be greater than or equal to zero, which means you don't have to check for negatives.
    

# Find Median from Data Stream

import heapq

class MedianFinder:
    def __init__(self):
        # Initialize two heaps: a max heap for the lower half and a min heap for the upper half
        self.lower = []  # Max heap
        self.upper = []  # Min heap

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        # Add to the appropriate heap
        if not self.lower or num <= -self.lower[0]:
            heapq.heappush(self.lower, -num)
        else:
            heapq.heappush(self.upper, num)

        # Balance the heaps
        if len(self.lower) > len(self.upper) + 1:
            heapq.heappush(self.upper, -heapq.heappop(self.lower))
        elif len(self.upper) > len(self.lower):
            heapq.heappush(self.lower, -heapq.heappop(self.upper))

    def findMedian(self):
        """
        :rtype: float
        """
        if len(self.lower) > len(self.upper):
            return -self.lower[0]
        return (-self.lower[0] + self.upper[0]) / 2.0
        
# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
    

# Word Ladder

class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        if endWord not in wordList:
            return 0
        
        # below lets you add empty list if key is new
        # nei for neighbor
        nei = collections.defaultdict(list)
        wordList.append(beginWord)

        # for each word
        for word in wordList:
            # look at each position in word
            for j in range(len(word)):
                pattern = word[:j] + "*" + word[j + 1:]
                nei[pattern].append(word)
        
        visit = set([beginWord])

        # BFS through the graph (usually best for finding shortest path!)
        q= deque([beginWord])

        res = 1
        # while q is nonempty 
        while q:
            for i in range(len(q)):
                word = q.popleft()
                if word == endWord:
                    return res
                # if not result, take neighbors and add to queue
                for j in range(len(word)):
                    pattern = word[:j] + "*" + word[j + 1:]
                    for neiWord in nei[pattern]:
                        if neiWord not in visit:
                            visit.add(neiWord)
                            q.append(neiWord)
            res += 1
        return 0


# Basic Calculator
    
class Solution(object):
    def calculate(self, s):
        stack = []
        operand = 0
        result = 0  # For the on-going result
        sign = 1  # 1 means positive, -1 means negative  

        for ch in s:
            if ch.isdigit():
                operand = (operand * 10) + int(ch)
            elif ch == '+':
                result += sign * operand
                sign = 1
                operand = 0
            elif ch == '-':
                result += sign * operand
                sign = -1
                operand = 0
            elif ch == '(':
                # Push the result and the sign onto the stack, for later
                stack.append(result)
                stack.append(sign)
                # Reset the sign and result for the new sub-expression
                sign = 1
                result = 0
            elif ch == ')':
                # Add whatever was left in the operand
                result += sign * operand
                # The result is now evaluated, apply the sign
                result *= stack.pop()  # stack pop 1, which is the sign before the parenthesis
                # Add to the next operand on the stack, which is the result calculated before this parenthesis
                result += stack.pop()  # stack pop 2
                operand = 0

        return result + sign * operand
    
# Maximum profit in job scheduling
    
from typing import List
import bisect

class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        # Combine the start, end time, and profit into one list and sort by end time
        jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])

        # dp array to store the maximum profit till each job
        dp = [0] * (len(jobs) + 1)

        for i in range(1, len(jobs) + 1):
            # Include current job profit
            incl_profit = jobs[i-1][2]
            # Find the latest job that doesn't conflict with the current job
            l = bisect.bisect_right(startTime, jobs[i-1][0], 0, i-1)
            incl_profit += dp[l]
            # Maximum profit is either including or excluding the current job
            dp[i] = max(dp[i-1], incl_profit)

        # The last element of dp will have the answer
        return dp[-1]


# Merge K sorted lists
    
# https://www.youtube.com/watch?v=q5a5OiGbT6Q

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:

        if not lists or len(lists) == 0:
            return None
        
        while len(lists) > 0:
            mergedLists = []

            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i+1] if (i+1) < len(lists) else None
                mergedLists.append(self.mergeList(l1,l2))
            lists = mergedLists
        return lists[0]

    
    def mergeList(self, l1, l2):
        dummy = ListNode()
        tail = dummy

        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        if l1:
            tail.next = l1
        if l2:
            tail.next = l2
        return dummy.next
    

import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists):
        if not lists:
            return None

        heap = []
        count = 0

        for l in lists:
            if l:
                heapq.heappush(heap, (l.val, count, l))
                count += 1

        dummy = ListNode(0)
        current = dummy

        while heap:
            _, _, node = heapq.heappop(heap)
            current.next = node
            current = current.next
            if node.next:
                heapq.heappush(heap, (node.next.val, count, node.next))
                count += 1

        return dummy.next

# Largest rectangle in histogram
# https://www.youtube.com/watch?v=zx5Sw9130L0

class Solution:
    def largestRectangleArea(self, heights):
        stack = [-1]
        max_area = 0

        for i in range(len(heights)):
            while stack[-1] != -1 and heights[stack[-1]] > heights[i]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)

        while stack[-1] != -1:
            h = heights[stack.pop()]
            w = len(heights) - stack[-1] - 1
            max_area = max(max_area, h * w)

        return max_area
