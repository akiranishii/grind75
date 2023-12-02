# Search in a rotated sorted array

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # https://www.youtube.com/watch?v=U8XENwh8Oy8
        left, right = 0, len(nums) - 1
        
        while left <= right:

            mid = (right + left) // 2

            if nums[mid] == target:
                return mid

            # Check if the left half is sorted
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Right half is sorted
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1
    

# Combination Sum

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(start, combination, total):
            if total == target:
                result.append(list(combination))
                return
            if total > target:
                return
            
            for i in range(start, len(candidates)):
                combination.append(candidates[i])
                backtrack(i, combination, total + candidates[i])
                combination.pop()
        
        result = []
        backtrack(0, [], 0)
        return result


# Alternative solution
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        dp = [[] for _ in range(target + 1)]
        dp[0] = [[]]  # Base case: one way to make zero sum

        for candidate in candidates:
            for current_sum in range(candidate, target + 1):
                for combination in dp[current_sum - candidate]:
                    dp[current_sum].append(combination + [candidate])

        return dp[target]

# Permutations

class Solution:
    # https://www.youtube.com/watch?v=DBLUa6ErLKw
    def __init__(self):
        self.res = []

    def permute(self, nums: List[int]) -> List[List[int]]:
        self.backtrack(nums,[])
        return self.res

    # nums is scope of nums that can be chosen from
    # path is each permutation
    def backtrack(self, nums, path):
        if not nums:
            self.res.append(path)
        for x in range(len(nums)):
            # nums includes everything except index in question
            # add index to path
            self.backtrack(nums[:x]+nums[x+1:], path+[nums[x]])


# Merge intervals

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # O(nlogn)
        intervals.sort(key = lambda interval : interval[0])
        output = [intervals[0]]

        for start, end in intervals[1:]:
            # get end value of most recently added interval to output
            lastEnd = output[-1][1]

            if start <= lastEnd: 
                output[-1][1] = max(lastEnd, end)

            else: 
                output.append([start, end])
        return output

# Lowest common ancestor of a binary tree

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None or root == p or root == q:
            return root

        # Recursively find LCA in the left and right subtrees
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        # If both left and right are non-null, this is the LCA
        if left and right:
            return root

        # If one of left or right is non-null, return the non-null one
        return left if left else right
    

# Time based key-value store

class TimeMap:

    def __init__(self):
        self.store = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.store:
            self.store[key] = []
        self.store[key].append([value, timestamp])

    def get(self, key: str, timestamp: int) -> str:
        res = ""
        values = self.store.get(key, [])

        # binary search
        l, r = 0, len(values) - 1
        while l <= r:
            m = (l + r)//2

            if values[m][1] <= timestamp:
                res = values[m][0]
                l = m + 1
            else:
                r = m - 1
    
        return res


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

# Accounts Merge

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x != self.parent.setdefault(x, x):
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        uf = UnionFind()
        email_to_name = {}

        # Step 1 and 2: Union emails and map emails to names
        for account in accounts:
            name = account[0]
            first_email = account[1]
            for email in account[1:]:
                uf.union(first_email, email)
                email_to_name[email] = name

        # Step 3: Group emails by their root
        email_groups = {}
        for email in email_to_name:
            root = uf.find(email)
            email_groups.setdefault(root, []).append(email)

        # Step 4: Sort and format the result
        return [[email_to_name[root]] + sorted(emails) for root, emails in email_groups.items()]



# Sort colors

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        low, mid, high = 0, 0, len(nums) - 1

        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1
            else:
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1
