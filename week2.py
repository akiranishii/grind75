
# First Bad Version
# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        left, right = 0, n
        while left < right:
            mid = (left+right)//2

            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left
    

# Ransom Note
# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        left, right = 0, n
        while left < right:
            mid = (left+right)//2

            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left

# better approaches: https://lifewithdata.com/2023/06/15/leetcode-ransom-note-solution-in-python/#google_vignette

# Climbing Stairs
class Solution:
    def climbStairs(self, n: int) -> int:
        # Number of ways to reach Step i = Number of ways to reach Step (i-1) + Number of ways to reach Step (i-2).

        if n == 1 or n == 2:
            return n

        twobefore = 1
        prev = 2 
        current = 0

        for step in range(3,n+1):
            current = twobefore + prev
            twobefore = prev
            prev = current

        return current
    

# Longest Palindrome
class Solution:
    def longestPalindrome(self, s: str) -> int:
        from collections import Counter

        char_dict = Counter(s)
        odd_exists = False
        length = 0

        for char_count in char_dict.values():
            if char_count % 2 == 0:
                length += char_count
            else:
                length += char_count - 1
                odd_exists = True

        if odd_exists == True:
            length += 1

        return length


# Reverse Linked List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        current = head

        while current:
            temp_next = current.next
            current.next = prev
            prev = current
            current = temp_next

        return prev
    
# majority element
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        from collections import Counter

        num_counter = Counter(nums)

        for num, count in num_counter.items():
            if count > len(nums)//2:
                return num
            

# More efficient solution:
# Boyer-Moore Voting Algorithm
# https://www.youtube.com/watch?v=n5QY3x_GNDg
#def majorityElement(nums):
#    candidate = None
#    count = 0

#    for num in nums:
#        if count == 0:
#            candidate = num
#        count += (1 if num == candidate else -1)

#   return candidate

# Add binary 
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        result = []
    
        # Start from the end of both strings
        i, j = len(a) - 1, len(b) - 1
        carry = 0
    
        # Loop through both strings
        while i >= 0 or j >= 0 or carry:
            total = carry
        
            if i >= 0:
                total += int(a[i])
                i -= 1
            
            if j >= 0:
                total += int(b[j])
                j -= 1
            
            # Append the sum of total % 2 to result
            result.append(str(total % 2))
            # Compute the carry
            carry = total // 2
    
        # Join the result and reverse it
        return ''.join(reversed(result))


# Diameter of Binary Tree
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # Initialize the maximum diameter
        self.max_diameter = 0
        
        def depth(node):
            # A null node has a depth of 0
            if not node:
                return 0
            
            # Recursively find the depth of the left and right subtrees
            left_depth = depth(node.left)
            right_depth = depth(node.right)
            
            # Update the maximum diameter (this step considers the node as the potential highest "root" of the diameter)
            self.max_diameter = max(self.max_diameter, left_depth + right_depth)
            
            # Return the depth of the tree rooted at this node (1 + max of the depths of the subtrees)
            return 1 + max(left_depth, right_depth)
        
        # Compute the depth of the tree and in the process, compute the diameter
        depth(root)
        
        # The max_diameter property now contains the result
        return self.max_diameter


# Middle of the Linked List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        return slow
    
    # Maximum Depth of Binary Tree
   # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        
        # at leaf node, max-depth is 0
        if not root:
            return 0


        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))


# Contains Duplicate
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        # Bad space complexity
        # return len(set(nums)) != len(nums)

        # Bad time complexity
        nums.sort()
        for i in range(len(nums) - 1):
            if nums[i] == nums[i + 1]:
                return True
        return False
    
# Maximum Subarray
class Solution:
    # https://www.youtube.com/watch?v=86CQq3pKSUw
    def maxSubArray(self, nums: List[int]) -> int:
        # Initialize our variables using the first element.
        current_subarray = max_subarray = nums[0]
        
        # Start with the second element since we already used the first one.
        for num in nums[1:]:
            # If current_subarray is negative, throw it away. Otherwise, keep adding to it.
            current_subarray = max(num, current_subarray + num)
            max_subarray = max(max_subarray, current_subarray)
        
        return max_subarray

