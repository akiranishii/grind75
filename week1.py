# Two Sum
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        
        seen = {}
        for i, num in enumerate(nums):
            remain = target - num

            if remain in seen:
                return [i, seen[remain]]

            seen[num] = i

# Valid Parentheses
class Solution:
    def isValid(self, s: str) -> bool:
        # ([{} ()])
        # whenever there is a closing bracket, it has to be the most recent opening bracket that has not been used! (stack)
        # The string cannot start with a closing bracket
        
        par_dict = {
            '(' : ')',
            '{': '}',
            '[': ']'
            }

        stack = []
        
        # check to make sure first string is not closing
        if s[0] in par_dict.values():
            return False

        # check to make sure last string is not opening
        if s[-1] in par_dict:
            return False

        for character in s:
            if character in par_dict:
                stack.append(character)
            else:
                
                if not stack:
                    return False

                last_opening = stack.pop()
                expected_closing = par_dict[last_opening]

                if character != expected_closing:
                    return False
            
        return not stack


# Merge two sorted lists
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = current = ListNode()

        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next
        
        current.next = list1 or list2

        return dummy.next


# Best time to buy and sell stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        left_index = 0
        profit = 0
        
        for right_index in range(1, len(prices)):

            if prices[left_index] < prices[right_index]:
                profit = max(profit, prices[right_index] - prices[left_index])
            else: 
                left_index = right_index
            
        
        return profit
    
# Valid Palindrome
class Solution:
    def isPalindrome(self, s: str) -> bool:
        # better to not create a new string due to space complexity issues
        #modified = s.lower()
        #modified = ''.join(c for c in modified if c.isalnum())

        #return modified == modified[::-1]

        left, right = 0, len(s) - 1 
 
        while left < right:
            while left < right and not s[left].isalnum():
                left +=1
            while right > left and not s[right].isalnum():
                right -= 1
            if s[left].lower() != s[right].lower():
                return False 
            left, right = left + 1, right - 1
        return True  

# Invert binary tree
class Solution:
    def isPalindrome(self, s: str) -> bool:
        # better to not create a new string due to space complexity issues
        #modified = s.lower()
        #modified = ''.join(c for c in modified if c.isalnum())

        #return modified == modified[::-1]

        left, right = 0, len(s) - 1 
 
        while left < right:
            while left < right and not s[left].isalnum():
                left +=1
            while right > left and not s[right].isalnum():
                right -= 1
            if s[left].lower() != s[right].lower():
                return False 
            left, right = left + 1, right - 1
        return True  


# Valid Anagram
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:

        if len(s) != len(t):
            return False

        dict_s = {}
        dict_t = {}

        for i in range(len(s)):
            
            # Update counts for string s
            if s[i] in dict_s:
                dict_s[s[i]] += 1
            else:
                dict_s[s[i]] = 1

            # Update counts for string t
            if t[i] in dict_t:
                dict_t[t[i]] += 1
            else:
                dict_t[t[i]] = 1

        # Check if the two dictionaries are the same
        return dict_s == dict_t

        # or you can sort them



# Binary search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (right+left) // 2

            if nums[mid] == target:
                return mid

            elif target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1

        
        return -1
    

# Flood fill
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        
        # Get the original color of the starting pixel
        origColor = image[sr][sc]
        
        # If the original color is already the same as newColor, no changes are needed
        if origColor == color:
            return image

        # Define the DFS function
        def dfs(r, c):
            # Base condition: check boundaries and if the pixel is of the same original color
            if r < 0 or r >= len(image) or c < 0 or c >= len(image[0]) or image[r][c] != origColor:
                return
            
            # Change the color of the pixel
            image[r][c] = color
            
            # Visit the 4-directionally connected neighbors
            dfs(r-1, c)  # Up
            dfs(r+1, c)  # Down
            dfs(r, c-1)  # Left
            dfs(r, c+1)  # Right
        
        # Start the DFS from the starting pixel
        dfs(sr, sc)
        
        return image
    
# Lowest common ancestor
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        curr = root

        while curr:
            if p.val < curr.val and q.val < curr.val:
                curr = curr.left
            elif p.val > curr.val and q.val > curr.val:
                curr = curr.right
            
            else: 
                return curr


# Balanced binary tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:

        self.answer = True

        def dfs(root):

            if not root:
                return 0
            
            l = 1 + dfs(root.left)
            r = 1 + dfs(root.right)

            if abs(l-r) > 1:
                self.answer = False

            return max(l,r)
        
        dfs(root)

        return self.answer
        

# Linked list cycle
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast, slow = head, head
        while fast and fast.next:
            fast, slow = fast.next.next, slow.next
            if fast == slow:
                return True
        return False
    

# Implement Queue using Stacks
class MyQueue:

    def __init__(self):
        self.push_stack = []
        self.pop_stack = []

    def push(self, x: int) -> None:
        self.push_stack.append(x)

    def pop(self) -> int:
        if self.empty(): return
        if len(self.pop_stack):
            return self.pop_stack.pop()
        else:
            while len(self.push_stack):
                self.pop_stack.append(self.push_stack.pop())
        return self.pop_stack.pop()

    def peek(self) -> int:
        if self.empty(): return
        if len(self.pop_stack):
            return self.pop_stack[-1]
        else:
            while len(self.push_stack):
                self.pop_stack.append(self.push_stack.pop())
        return self.pop_stack[-1]
    

    def empty(self) -> bool:
        return len(self.push_stack)==False and len(self.pop_stack) == False
    