# 数组

1. 704 二分查找 [704. 二分查找 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-search/)
   有序数组，while与双指针找区域

   ```python
   class Solution:
       def search(self, nums: List[int], target: int) -> int:
           left = 0
           right = len(nums) - 1 # 左闭右闭，均可以到达
           while left <= right:
               if nums[(left + right)//2] < target:
                   # 在中间值的索引加减1，因为显然中间值已经不等
                   left =  (left + right)//2 +1
               elif nums[(left + right)//2] > target:
                   right = (left + right)//2 -1
               else:
                   return (left + right)//2
           
           return -1
   ```

2. 27移除元素  [27. 移除元素 - 力扣（LeetCode）](https://leetcode.cn/problems/remove-element/description/)
   原地，使用双指针的方法，**所以是通当前元素不等做判断的，不等就赋值slow。否则继续等**

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        slow = 0
        fast = 0
        cnt = 0

        # for fast in range(len(nums)):
        #     # 遇到元素不是，就把slow赋值fast的索引元素，然后slow加1
        #     # 否则fast继续往前走，slow留在下一个能够选的位置
        #     if nums[fast] != val:
        #         nums[slow] = nums[fast]
        #         slow += 1
        #         cnt += 1
        # return cnt
        
        # 下面这里用while
        while fast < len(nums):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
                cnt += 1
            fast += 1
        
        return cnt
            
```

3. 有序数组的平方 [977. 有序数组的平方 - 力扣（LeetCode）](https://leetcode.cn/problems/squares-of-a-sorted-array/)
   ```python
   class Solution:
       def sortedSquares(self, nums: List[int]) -> List[int]:
           # 自定义库
           # for i in range(len(nums)):
           #     nums[i] = nums[i] ** 2
           # nums.sort()
           # return nums
   
           # 双指针
           # 通过比较两段的结果谁大，存进最大的位置
           left = 0
           i = len(nums) -1
           right = len(nums) -1
           res = [0] * len(nums)
           while left <= right: # 指针相遇结束
               print(i)
               num_left = nums[left] ** 2
               num_right = nums[right] ** 2
   
               if num_left >= num_right:
                   res[i] = num_left
                   # 左指针前进
                   left += 1
               else:
                   res[i] = num_right
                   # 右指针左移
                   right -= 1
               # 存放指针移动
               i -= 1
           return res
   ```

   

# 链表

单链表：head-node-null, node: next, val
双链表：head-node-null，null-node-tail，node:prev,next, val
循环链表：head-node-head

> ```python
> class ListNode:
>     def __init__(self, val, next=None):
>         self.val = val
>         self.next = next
> ```

1.  [203. 移除链表元素 - 力扣（LeetCode）](https://leetcode.cn/problems/remove-linked-list-elements/)
   ```python
   class Solution:
       def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
           # 创建dummy 保留head
           dummy = ListNode(next=head)
   
           # 用cur来遍历链表
           cur = dummy
           while cur.next: # 下一个节点存在
               if cur.next.val == val:
                   cur.next = cur.next.next
               else:
                   cur = cur.next
           return dummy.next
   ```

2. [707. 设计链表 - 力扣（LeetCode）](https://leetcode.cn/problems/design-linked-list/description/)

3. 反转链表 [206. 反转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-linked-list/)
   ```python
   class Solution:
       def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
           cur = head
           prev = None
   
           # 用这个节点存在就反向指针，避免了最后一个None
           while cur:
               nxt = cur.next
               # 反向指针
               cur.next = prev
               # 当前节点作为下一轮的prev
               prev = cur
               # 下一个节点（不存储结果）
               cur = nxt
           
           return prev
   ```

   4. 两两交换 [24. 两两交换链表中的节点 - 力扣（LeetCode）](https://leetcode.cn/problems/swap-nodes-in-pairs/)
      ```python
      class Solution:
          def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
              dummy = ListNode(next=head)
              cur = dummy
      
              # 使用cur遍历整个链表
              # 因为是以两个节点为最小元素操作，需要需要判断是否有两个元素
              while cur.next and cur.next.next:
                  nxt = cur.next
                  nnnxt = cur.next.next.next
      
                  # 然后执行一次翻转
                  # 下一个元素指向下下个元素
                  cur.next = cur.next.next
                  # 然后下下个元素指向下一个元素
                  cur.next.next = nxt
      
                  # 准备下一轮循环！
                  # 然后下一个元素要指向下一周期的第一个元素
                  nxt.next = nnnxt
                  # 下一个节点是这个周期第二个元素
                  cur = nxt
              
              return dummy.next
          
          # 没学会！
          # 最小元素使用递归完成
              if not head or not head.next:
                  return head
              
              prev = head
              cur = head.next
              nxt = head.next.next
      
              # 下一个元素指向下下个元素
              cur.next = prev
              # 下下个元素指向下一个
              prev.next = self.swapPairs(nxt)
              
              return cur
      ```

      

# 贪心

**贪心的本质是选择每一阶段的局部最优，从而达到全局最优**，一般是需要先考虑当前问题中：

- 局部最有能够达到全局最优



3. 摆动序列https://leetcode.cn/problems/wiggle-subsequence/
   ```python
   class Solution:
       def wiggleMaxLength(self, nums: List[int]) -> int:
           if len(nums) == 2:
               return 1 if nums[0] == nums[1] else 2
   
           cnt = 1 # 从1开始考虑，因为头没有数据
           past_diff = 0
           for i in range(1,len(nums)):
               cur_diff = nums[i] - nums[i-1]
               if (past_diff<=0 and cur_diff>0) or (past_diff>=0 and cur_diff<0):
                   # 记住这里的cur_diff，不能等于0的，这样等于0说明这里是平坡，没有升降
                   cnt += 1
                   # 要考虑单调平坡的情况，会导致结果多一个
                   past_diff = cur_diff
   
           return cnt
   ```

4. 最大子数组和https://leetcode.cn/problems/maximum-subarray/
   ```python
   class Solution:
       def maxSubArray(self, nums: List[int]) -> int:
           # 如果当前子序列结果为负，加上正的也是损失，所以当子序列之和为负的时候，就将下一个元素作为新的序列头
           res = float('-inf')
           sum_ = 0
           for i in range(len(nums)):
               sum_ += nums[i]
               res = max(sum_, res)
               if sum_ < 0 :
                   # 重新算就是下一个元素作为开头
                   sum_ = 0 
               
           return res
   ```

5. 买卖股票的最佳时机https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/
   ```python
   class Solution:
       def maxProfit(self, prices: List[int]) -> int:
           # 转化为计算前缀差，大于0就是利润，小于0就是亏损
           # 然后计算利润和
           res = 0
           for i in range(1,len(prices)):
               tmp = prices[i] - prices[i-1]
               res += max(0, tmp) # 神
           return res
   ```

   

6. 跳跃游戏https://leetcode.cn/problems/jump-game/
   ```python
   class Solution:
       def canJump(self, nums: List[int]) -> bool:
           if len(nums)==1: 
               return True
           cover = 0
           for i in range(len(nums)-1):
               if i <= cover: # 如果当前位置在可覆盖范围内
                   cover = max(i+nums[i], cover) # 更新能覆盖的最远位置
                   if cover >= len(nums)-1 : return True
           
           return False
   ```

7. 跳跃游戏II[45. 跳跃游戏 II - 力扣（LeetCode）](https://leetcode.cn/problems/jump-game-ii/)

```python
class Solution:
    def jump(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return 0 

        cur_dst = 0 # 用于记录当前的最大范围
        nxt_dst = 0 # 下一步的最大范围
        steps = 0

        for i in range(len(nums)):
            nxt_dst = max(i+nums[i], nxt_dst) # 更新下一步最大范围
            if i == cur_dst: #走到了上一步的最大距离
                steps += 1 #该走下一步
                cur_dst = nxt_dst #期间的下一步最大距离赋值给最大距离
                if nxt_dst >= len(nums)-1:
                    break
        
        return steps
```

8. K次取反后最大化的数组和[1005. K 次取反后最大化的数组和 - 力扣（LeetCode）](https://leetcode.cn/problems/maximize-sum-of-array-after-k-negations/)
   ```python
   class Solution:
       def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
           nums.sort()
           for i in range(k):
               # 每一步都把最小值变成相反数
               nums[0] = nums[0] * (-1)
               nums.sort()
           
           return sum(nums)
   ```

9.加油站[134. 加油站 - 力扣（LeetCode）](https://leetcode.cn/problems/gas-station/)

**for循环适合模拟从头到尾的遍历，而while循环适合模拟环形遍历，要善于使用while！**

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        rest = 0 # 统计当前油箱还剩下多少
        minFuel = float('inf') # 跑完全程中邮箱里面的最小油量

        for i in range(len(gas)):
            rest += gas[i] - cost[i]
            minFuel = min(rest, minFuel)
        
        # 第一种情况，剩余油量小于0
        if rest < 0:
            return -1
        
        # 可以到达，但是需要判断从哪里出发

        # 第二种情况，从起点出发到任何一个加油站时最小油量都大于等于0，从头
        if minFuel >= 0:
            return 0
        
        # 第三种情况，从后往前，到哪个站的盈余能够补上最小油量，就从那里出发
        # 从后往前转，因为中间有一个站净剩余复数很大，前面都不能跑
        for i in range(len(gas)-1, -1, -1):
            minFuel += gas[i] - cost[i]
            if minFuel >= 0:
                return i
           
# 法二
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        rest = 0 # 统计当前油箱还剩下多少
        totalRest = 0 #最后剩多少
        strat_idx = 0

        for i in range(len(gas)):
            rest += gas[i] - cost[i]
            totalRest += gas[i] - cost[i]

            if rest < 0:
                # 如果当前的剩余小于0，说明这一段不对
                # 类似于最大子序列和
                strat_idx = i+1
                rest = 0
        
        if totalRest < 0:
            return -1
        else:
            return strat_idx
```

10. 分发糖果[135. 分发糖果 - 力扣（LeetCode）](https://leetcode.cn/problems/candy/description/)
    ```python
    class Solution:
        def candy(self, ratings: List[int]) -> int:
            candy_l = [1]
            candy = 1
            # 从左向右，右边比左边大就加1， 否则就只给1个（体现贪心）
            for i in range(1, len(ratings)):
                if ratings[i] >ratings[i-1]:
                    candy += 1
                    candy_l.append(candy)
                else:
                    candy = 1 
                    candy_l.append(candy)
    
            # 从右向左
            candy_r = [1]
            candy = 1
            for i in range(len(ratings)-2 , -1 ,-1):
                if ratings[i] >ratings[i+1]:
                    candy += 1
                    candy_r.append(candy)
                else:
                    candy = 1
                    candy_r.append(candy)
            # 记得倒序
            candy_r = candy_r[::-1]
    
            # 兼顾两边，冲突就给大，灵魂
            res = 0
            for i in range(0, len(candy_l)):
                # print(max(candy_l[i], candy_r[i]))
                res+= max(candy_l[i], candy_r[i])
            return res
        
        # 法二 
        class Solution:
        def candy(self, ratings: List[int]) -> int:
            n = len(ratings)
            candy_l = [1] *n
            # 从左向右，右边比左边大就加1， 否则就只给1个（体现贪心）
            for i in range(1, len(ratings)):
                if ratings[i] >ratings[i-1]:
                    candy_l[i] = candy_l[i-1] + 1
    
            # 从右向左
            candy_r = [1]*n
            for i in range(len(ratings)-2 , -1 ,-1):
                if ratings[i] >ratings[i+1]:
                    candy_r[i] = candy_r[i+1] + 1
    
            # 兼顾两边，冲突就给大，灵魂
            res = 0
            for i in range(0, len(candy_l)):
                # print(max(candy_l[i], candy_r[i]))
                res+= max(candy_l[i], candy_r[i])
            return res
        
    ```

    

11. 柠檬水找零
    这道题目可以告诉大家，遇到感觉没有思路的题目，**可以静下心来把能遇到的情况分析一下，只要分析到具体情况了**，一下子就豁然开朗了。
    这道题不是比较余额，而是比较不同面值的钞票能否找零。

    **从下面使用{}和 defaultdict的用法能够看出，后者更加方便，在初始的时候就默认赋值为0，而前者则需要在操作中生成key**

    ```python
    class Solution:
        def lemonadeChange(self, bills: List[int]) -> bool:
            if bills[0] != 5:
                return False
            res = {}
            res = defaultdict(int)
            for bill in bills:
                if bill == 5:
                    # 没有就先赋值0，否则加1
                    res['5'] = res.get('5', 0) + 1 
                elif bill == 10:
                    res['10'] = res.get('10', 0) + 1 
                    res['5'] -= 1
                    # 给不出找零
                    if res['5'] < 0:
                        return False
                elif bill == 20:
                    res['20'] = res.get('20', 0) + 1 
                    # 优先给10元钞票，因为5元的用处更灵活
                    if res.get('10', 0) > 0:
                        res['10'] = res.get('10', 0) - 1 
                        res['5'] -= 1
                    else:
                        res['5'] -= 3
                    # 给不出找零
                    if res['5'] < 0 or res['10'] < 0:
                        return False
            return True
    
    
    from collections import defaultdict
    class Solution:
        def lemonadeChange(self, bills: List[int]) -> bool:
            if bills[0] != 5:  # 第一个顾客必须给5元
                return False
            
            res = defaultdict(int)
            for bill in bills:
                if bill == 5:
                    res[5] += 1  # 直接增加计数
                elif bill == 10:
                    res[10] += 1
                    res[5] -= 1
                    if res[5] < 0:
                        return False
                elif bill == 20:
                    # 优先使用10+5的组合找零
                    if res[10] > 0:
                        res[10] -= 1
                        res[5] -= 1
                    else:
                        # 只能用3张5元找零
                        res[5] -= 3
                    
                    if res[5] < 0:
                        return False
            return True
    ```

12. 根据身高重建队列[406. 根据身高重建队列 - 力扣（LeetCode）](https://leetcode.cn/problems/queue-reconstruction-by-height/submissions/639958325/)

**本题有两个维度，h和k，看到这种题目一定要想如何确定一个维度，然后再按照另一个维度重新排列。**

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        # 先按照h排序，然后按照k来插入之前的排序
        # 因为身高排序之后，后面的人插入不会影响排序
        # 而具体插入到哪里，直接按k来决定就行

        # 按照索引排序，身高的降序，人数的升序
        people.sort(key=lambda x:(-x[0], x[1]))
        print(people) 

        # 用人数来调整
        res = []
        for p in people:
            res.insert(p[1], p)
        return res
```

13. 452[452. 用最少数量的箭引爆气球 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/)

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        # 先按照左边界排序
        points.sort(key=lambda x:x[0])
        print(points)
        res = 1

        # 现在遍历所有的气球，如果下一个气球左边界超过现在气球右边界，就增加一根箭
        for i in range(1, len(points)):
            if points[i][0] > points[i-1][1]:
                res += 1
            else:
                # 重叠气球，但是要取交叠部分的右边界作为新的边界
                points[i][1] = min(points[i][1], points[i-1][1])
                  
        return res
```

14. 435[435. 无重叠区间 - 力扣（LeetCode）](https://leetcode.cn/problems/non-overlapping-intervals/description/)

    ```python
    class Solution:
        def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
            # 类似于上一题，用了几根箭说明存在几加1个交叠区域，把
            intervals.sort(key=lambda x:x[0])
            print(intervals)
    
            cnt = 0 # 交叠区域
            for i in range(1, len(intervals)):
                # 不用取等，因为这里相等不会产生交叠
                if intervals[i][0] < intervals[i-1][1]:
                    # 出现了交叠
                    cnt += 1
                    intervals[i][1] = min(intervals[i][1], intervals[i-1][1])
            return cnt
    ```

    

15. 763[763. 划分字母区间 - 力扣（LeetCode）](https://leetcode.cn/problems/partition-labels/description/)

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        # 用于存储每个字母出现的左右边界
        hash_ = [[-1, -1] for _ in range(26)]

        # 统计
        for i in range(len(s)):
            key = ord(s[i]) - ord('a')
            if hash_[key][0] == -1: # 这里不能以0作为判断条件，因为第一个字母的位置会冲突
                hash_[key][0] = i 
            hash_[key][1] = i

        # 排除没出现过的字母
        points = []
        for ele in hash_:
            if ele != [-1, -1]:
                points.append(ele)
        
        # 找无交叠区间，但是有一点区别，就是按照交叠区域的最大右边界来划分
        points.sort(key=lambda x:x[0])
        res = []
        boundary_r = points[0][1]
        boundary_l = points[0][0]

        for i in range(1, len(points)):
            # 出现离群点
            if points[i][0] > boundary_r:
               res.append(boundary_r - boundary_l + 1)
               boundary_l = points[i][0]
            # 没出现离群点，但是需要更新右边界
            boundary_r = max(boundary_r, points[i][1])
        
        # 还剩下一块，统计长度
        res.append(boundary_r - boundary_l + 1)
        return res
```

16. 56 合并区间[56. 合并区间 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-intervals/)

    ```python
    class Solution:
        def merge(self, intervals: List[List[int]]) -> List[List[int]]:
            if len(intervals) <= 1:
                return intervals
            intervals.sort(key=lambda x:x[0])
            res = []
            boundar_l = intervals[0][0]
            boundar_r = intervals[0][1]
    
            for i in range(1, len(intervals)):
                # 区域交叠就合并
                if intervals[i][0] <= boundar_r:
                    boundar_l = min(intervals[i][0], boundar_l)
                    boundar_r = max(intervals[i][1], boundar_r)
                else:
                    res.append([boundar_l, boundar_r])
                    boundar_l = intervals[i][0]
                    boundar_r = intervals[i][1]
            res.append([boundar_l, boundar_r])
            return res
    
    ```

    

17. 738 单调递增的数字[738. 单调递增的数字 - 力扣（LeetCode）](https://leetcode.cn/problems/monotone-increasing-digits/description/)
    数字文本的更改通过上下切分，单独拿出的方式完成，例如更改str中的第i-1个位置：

    >  strNum = strNum[:i-1] + str(int(strNum[i-1]) - 1) + strNum[i:]

    ```python
    class Solution:
        def monotoneIncreasingDigits(self, n: int) -> int:
            # 如果每个去比较，会出现不是最大的问题
            # 通过找不满足条件的最高位-1， 后面的都变成9，现在任务变成找不是递增的最高位
            n_in = str(n)
            flag = len(n_in)
    
            for i in range(len(n_in)-1, 0, -1): 
                if int(n_in[i-1]) > int(n_in[i]):
                    flag = i #只会减小，不用取小
                    # 前一个位置减小1
                    n_in = n_in[:i-1] + str(int(n_in[i-1])-1) + n_in[i:]
            
            for i in range(flag, len(n_in)):
                n_in = n_in[:i] + '9' + n_in[i+1:]
            
            return int(n_in)
            
    ```

18. 968监控二叉树 [968. 监控二叉树 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-tree-cameras/)
    对于递归的时候，需要持续记录并且返回的参数一般有三种方法：

    - 声明全局变量，函数外部设置变量，内部声明nonlocal variable
    - 创建可变变量，如list:[0]，设置变量为一个list的元素，函数内部可以更改
    - 使用类的属性，设置self.value = 0

    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def minCameraCover(self, root: Optional[TreeNode]) -> int:
            # 安装摄像头下，每个节点的状态
            # 0：该节点有摄像头；没摄像头时，1：该节点被覆盖，或者2：该节点没被覆盖 
    
            # 贪心：从下往上安装摄像头，其中叶子节点不安装，因为最多覆盖一个，所以安装叶子节点的父节点
            
            # res = [0]
            res = 0 
            def dfs(node):
                nonlocal res
                if not node:
                    return 1
                
                # 先遍历子节点，然后根据子节点情况设置本节点
                left = dfs(node.left)
                right = dfs(node.right)
    
                # 情况一：左右节点都有覆盖，由于自下向上，所以本节点没被覆盖
                if left == 1 and right == 1:
                    return 2
                
                # 情况二：左右节点有一个没被覆盖，那么本节点需要安装一个摄像头
                if left == 2 or right == 2:
                    res += 1
                    return 0
                
                # 情况三：左右节点有一个装了摄像头，本节点有覆盖
                if left == 0 or right == 0:
                    return 1
            
            # 递归到头结点， 如果头结点是0
            if dfs(root) == 2:
                res += 1
    
            return res 
    ```

    

# 动态规划

## 动态规划五部曲

- dp数组以及下标的意义：dp[i], dp\[i]\[j]
- 递归公式
- dp数组初始化：dp[0]
- 遍历顺序：从后往前还是从前往后
- 打印数组检查

## 思考

递归最大的思想就是遍历的同时保存结果，然后后面的计算在先前的计算基础上完成，从而减少计算重复，所以递归最重要的是从小的元素思考，然后想如何使用这些结果进行状态转移

dp数组有三种：

- 一维dp
  在题目的条件中，有的是有条件的状态转移，例如nums[j] < nums[i]

- 二维dp

- 一维dp加上额外状态
  不同状态之间有条件转移

  

## 题目

1. 503 斐波那契数列

   ```python
   class Solution:
       def fib(self, n: int) -> int:
           # 第一步，下标意义，dp[i]是位置i的数值
           # 第二步，递推公式，dp[i] = dp[i-1] + dp[i-2]
           # 第三步，初始化，dp[0] = 1, dp[1] = 1
           # 第四步，遍历顺序，从前往后
           # 第五步，打印结果检查
           dp = [1] * n
           for i in range(2,n):
               dp[i] = dp[i-1] + dp[i-2]
   
           print(dp)
           return dp[-1] if dp else 0
   ```

   

2. 70 爬楼梯

   ```python
   class Solution:
       def climbStairs(self, n: int) -> int:
           # dp[i]是到第i层阶梯的方法数
           # dp[i] = dp[i-1] + dp[i-2]，因为到i层只能从i-1层走1步，或者是从i-2层走两步
           # dp[1] = 1; dp[2] = 2
           # 从前往后遍历
           # print
   
           if n == 1:
               return 1
           dp = [0]*n
           dp[0] = 1
           dp[1] = 2
           for i in range(2,n):
               dp[i] = dp[i-1] + dp[i-2]
           print(dp)
           return dp[-1]
   ```

3. 最小花费爬楼梯

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # 站在那一层不收费，但是往上跳收费

        # dp[i]代表了到达这一层需要花费的金额
        # dp[i] = min{dp[i-1]+cost[i-1], dp[i-2]+cost[i-2]}
        # dp[0] = cost[0], dp[1] = cost[1]
        # 从一开始往后遍历
        # 打印数据

        dp = [0] * (len(cost)+1)
        dp[0] = 0
        dp[1] = 0

        for i in range(2, len(cost)+1):
            dp[i] = min(dp[i-1] + cost[i-1], dp[i-2]+cost[i-2])
        
        print(dp)
        return dp[-1]
```

4. 62不同路径 [62. 不同路径 - 力扣（LeetCode）](https://leetcode.cn/problems/unique-paths/)

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # dp[i][j]表示在从原点到[i, j]的路径数目
        # dp[i][j] = dp[i-1][j] + dp[i][j-1] 当前节点只能从上面和做面两个节点到达
        # 初始化dp[i][0]，dp[0][j] =1
        # 从前往后
        
        matrix = [[0 for _ in range(n)] for _ in range(m)] 
        for i in range(m): matrix[i][0] = 1
        for i in range(n): matrix[0][i] = 1  
        
        for row in range(1, m):
            for col in range(1, n):
                matrix[row][col] = matrix[row-1][col] + matrix[row][col-1]
        
        print(matrix)
        return matrix[m-1][n-1]
```

5. 63不同路径II [63. 不同路径 II - 力扣（LeetCode）](https://leetcode.cn/problems/unique-paths-ii/)

   ```python
   class Solution:
       def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
           # 思路，在递推的过程中，如果相应位置出现了时候，就把路径设置为0
           rows, cols = len(obstacleGrid), len(obstacleGrid[0])
   
           matrix = [[0 for _ in range(cols)] for _ in range(rows)]
           # 障碍物之后的格点应该是0
           for row in range(rows): 
               if obstacleGrid[row][0] == 0:
                   matrix[row][0] = 1  
               else:
                   break
   
           for col in range(cols): 
               if obstacleGrid[0][col] == 0:
                   matrix[0][col] = 1  
               else: 
                   break
   
           # 有障碍物的地方不计入当前格点路径数
           for row in range(1, rows):
               for col in range(1, cols):
                   if obstacleGrid[row][col] == 0:
                       matrix[row][col] = matrix[row-1][col] + matrix[row][col-1] 
           
           # print(matrix)
           return matrix[rows-1][cols-1]
   ```

6. 整数拆分 [343. 整数拆分 - 力扣（LeetCode）](https://leetcode.cn/problems/integer-break/)

   ```python
   class Solution:
       def integerBreak(self, n: int) -> int:
           # dp[i]代表了i整数拆分的最大值
           # 递推关系：dp[i] = max(j*dp[i-j], j*(i-j), dp[i])
           # dp[0] = 0, dp[1] = 0, dp[2]=1
           # 从前往后遍历
           if n <2:
               return 0
           
           # 需要存储n+1个数， 因为n=2,需要算dp[3]
           dp = [0] * (n+1)
           dp[0] = 0
           dp[1] = 0
           dp[2] = 1
   
           for i in range(3, n+1):
               # 对拆下来的数进行试探
               for j in range(1, i):
                   # 是否继续拆分下去
                   dp[i] = max(j*dp[i-j], j*(i-j), dp[i])
   
           print(dp)
           return dp[n]
   ```

7. 96不同的二叉搜索树 [96. 不同的二叉搜索树 - 力扣（LeetCode）](https://leetcode.cn/problems/unique-binary-search-trees/)

```python
class Solution:
    def numTrees(self, n: int) -> int:
        # 二叉搜索树是指左子节点小于父节点，父节点小于右子节点
        # dp[i] 是i个节点所构成的二叉搜索树的所有可能性
        # dp[i] = dp[1]*dp[i-1] + dp[2]*dp[i-2] + ···
        # dp[0] = 1, dp[1] = 1, dp[2] = 2,
        # 从小往大

        if n < 2:
            return 1
        dp = [0] * (n+1)
        dp[0] = 1
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n+1):
            for j in range(0, i+1):
                # 右边的两个因子分别代表了，把j拿出来当根节点之后，比j小的j-1个元素构成的左子树和比j大的i-j个元素构成的右子树
                dp[i] += dp[j-1] * dp[i-j]
        
        return dp[n]

```

8. 0/1背包问题 [46. 携带研究材料（第六期模拟笔试）](https://kamacoder.com/problempage.php?pid=1046)

   先遍历物体，再遍历背包，并且倒序遍历(bag, weight[i]-1, -1)

   ```python
   num_obj, bag = map(int, input().split())
   spaces = list(map(int, input().split()))
   values = list(map(int, input().split()))
   
   # num_obj = 6 
   # bag = 3
   # spaces = [2, 2, 3, 1, 5, 2]
   # values = [2, 3, 1, 5, 4, 3]
   
   # dp[i][j]是从0-i个物体中装满空间背包j的最大价值
   # 递推公式就是装还是不装下一个重量为spaces[i]的物体
   # dp[i][j] = max(dp[i-1][j-spaces[j]] + values[i], dp[i-1][j])
   # dp[i][0] = 0, dp[0][j]中，j如果超过第一个物体重量就是values[0],否则0
   
   dp = [[0] * (bag+1) for _ in range(num_obj)]
   for col in range(spaces[0], bag+1):
       dp[0][col] = values[0] 
   
   for row in range(1, num_obj): #遍历物体
       for space in range(bag+1):
           if spaces[row] <= space:
               dp[row][space] = max(dp[row-1][space-spaces[row]] + values[row], dp[row-1][space])
           else:
               dp[row][space] = dp[row-1][space]
   
   print(dp[num_obj-1][bag])
   ```

   

9. 0/1背包II [46. 携带研究材料（第六期模拟笔试）](https://kamacoder.com/problempage.php?pid=1046)
   ```python
   num_obj, bag = map(int, input().split())
   spaces = list(map(int, input().split()))
   values = list(map(int, input().split()))
   
   # num_obj = 6 
   # bag = 3
   # spaces = [2, 2, 3, 1, 5, 2]
   # values = [2, 3, 1, 5, 4, 3]
   
   # dp[j]是装满空间背包j的最大价值
   # 递推公式就是装还是不装下一个重量为spaces[i]的物体
   # dp[j] = max(dp[j-spaces[j]] + values[i], dp[j])
   # dp[0] = 0,
   
   dp = [0] * (bag+1)
   for i in range(num_obj): #先遍历物体
       for j in range(bag, spaces[i]-1, -1): # 不断地走向下一个物体
           dp[j] = max(dp[j], dp[j-spaces[i]] + values[i])
   
   print(dp[bag])
   
   ```

10. 416 分割等和子集

    ```python
    class Solution:
        def canPartition(self, nums: List[int]) -> bool:
            # 从0/1背包的思路进行迁移，将list内部的数值累加，除以2
            # 如果将每个元素的数值视作其重量和价值，那么问题就转化为
            # 是否可以将sum//2的背包装满，并且它的价值刚好就是sum//2
    
            # 剪枝
            if sum(nums)%2 != 0:
                return False
    
            bag = sum(nums)//2
            dp = [0]*(bag+1)
    
            for i in range(len(nums)):
                for j in range(bag, nums[i]-1, -1): # nums[i]-1是因为for循环函数的右边是开的
                    if nums[i] <= bag:
                        dp[j] = max(dp[j], dp[j-nums[i]] + nums[i])
                    else:
                        dp[j] = dp[j-1]
            print(dp)
            return True if dp[bag] == bag else False
    ```

    

11. 1049 最后一块石头的重量II

    ```python
    class Solution:
        def lastStoneWeightII(self, stones: List[int]) -> int:
            bag = sum(stones)//2
            dp = [0] * (bag+1)
    
            for i in range(len(stones)):
                for j in range(bag, stones[i]-1, -1):
                    if stones[i] <= bag:
                        dp[j] = max(dp[j], dp[j-stones[i]]+stones[i])
                    else:
                        dp[j] = dp[j-1]
    
            # 总重量减去两倍的最接近总重量一半的重量
            total_sum = sum(stones)
            total_sum -= 2*dp[bag]
            return total_sum
    ```

12. 目标和
    ```python
    class Solution:
        def findTargetSumWays(self, nums: List[int], target: int) -> int:
            # left + right = sum
            # left - right = target
            # left = (sum + target) // 2
    
            # 那么现在的问题就变成了是否能够将背包装满left重量的物体
            # 背包问题提供了一种方案：能够刚好将物体按照剩余空间转进背包
    
            bag = (sum(nums) + target) // 2
            dp = [0] * (bag+1)
            for i in range(len(nums)):
                for j in range(bag, nums[i]-1, -1):
                    if nums[i] <= bag:
                        dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])
                    else:
                        dp[j] = dp[j-1]
            print(dp)
    ```

    

13. 474 一和零https://leetcode.cn/problems/ones-and-zeroes

    ```python
    class Solution:
        def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
            dp = [[0]*(n+1) for _ in range(m+1)] # 代表了背包重量
            for s in strs:
                # 0/1统计,python提供string的计数方法
                zeros = s.count('0')
                ones = s.count('1') 
                
                # 这个地方没有特别懂
                for j1 in range(m, zeros-1, -1):
                    for j2 in range(n, ones-1, -1):
                        dp[j1][j2] = max(dp[j1][j2], dp[j1-zeros][j2-ones]+1)
            
            return dp[m][n] 
    ```

14. 完全背包 [52. 携带研究材料（第七期模拟笔试）](https://kamacoder.com/problempage.php?pid=1052)
    **完全背包问题解决的是装满这个背包的最佳价值，或者是能不能装满这个背包**

    ```python
    n, bag_w = map(int, input().split())
    weights = [0] * n
    values = [0] * n
    
    for i in range(n):
        weights[i], values[i] = map(int, input().split())
    
    # 二维dp，因为无法重新遍历
    dp = [[0] * (bag_w+1) for _ in range(n)]
    
    # 先遍历物品，再遍历背包
    for i in range(n):
        # 这里需要正序遍历，因为要保证可重复填小物品
        for j in range(bag_w+1):
            if weights[i] <= j:
                # dp[i][j] = max(dp[i-1][j], dp[i][j-weights[i]] + values[i])，不会减少物品选择
                dp[i][j] = max(dp[i-1][j], dp[i][j-weights[i]] + values[i])
            else:
                # 这里不变
                dp[i][j] = dp[i-1][j]
    
    print(dp[n-1][bag_w])
    ```

15. 518 零钱兑换 

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        # dp[j]代表凑够金额为j的方案
        # dp[j] += dp[j-i] # i是金额的选择数目
        # dp[0] = 1初始化为1，否则后面结果累加都是1； 而非零下标都标记为0，否则影响结果

        dp = [0] * (amount+1)
        dp[0] = 1

        #先遍历物品
        for i in range(len(coins)):
            for j in range(amount+1): #这里一定要amount+1,否则会漏掉最后的amount
                print(j)
                print(dp)
                if j >= coins[i]:
                    dp[j] += dp[j-coins[i]]

        # print(dp)
        return dp[amount]
```



16. 组合综合IV
    **这道题其实要求了物体的顺序，所以在遍历的时候需要先遍历背包容量，在遍历物体**

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        # 和上一题类似，但是这一题要求顺序了
        dp = [0] * (target+1)
        dp[0] = 1

        # 在求排列而非组合的时候需要对背包进行遍历，
        # 如果先对物品遍历，无形中就会对物体进行排序，从而缺失一些结果
        for j in range(target+1):
            for i in range(len(nums)):
                if j >= nums[i]:
                    dp[j] += dp[j-nums[i]]
        
        # print(dp[target])
        return dp[target]
```

17. 爬楼梯，跳多层
    ```python
    n, m = map(int, input().split())
    
    # m其实就是物品， n就是背包容量，要学会抽象
    # dp[j]代表了上j层楼梯的方案数目
    # 所以dp[j] += dp[j-i]
    dp = [0] * (n+1)
    dp[0] = 1
    
    # 而且这题显然不是组合，而是排列！！
    # 这里i的遍历顺序是(1, m+1)，要结合题意与for循环规则
    for j in range(1, n+1):
        for i in range(1, m+1):
            if j >= i:
                dp[j] += dp[j-i]
    
    print(dp[n])
    ```

    18. 322 零钱兑换

        ```python
        class Solution:
            def coinChange(self, coins: List[int], amount: int) -> int:
                # dp[j]代表了凑成金额j的最小硬币个数
                # dp[j] = min(dp[j], dp[j-i]+1)
                # dp[0] = 0
        
                dp = [float('inf')] * (amount+1)
                dp[0] = 0
                
                for i in range(len(coins)):
                    for j in range(amount+1):
                        # 这里还需要判断dp[j-coin[i]]初始值，不是才进行状态转移
                        # j >= coins[i]
                        if j >= coins[i] and dp[j-coins[i]] != float('inf'):
                            dp[j] = min(dp[j], dp[j-coins[i]]+1)
        
                return dp[amount] if dp[amount] != float('inf') else -1
        ```

        

18. 279 完全平方数

    ```python
    class Solution:
        def numSquares(self, n: int) -> int:
            # dp[j]是凑成n的最小平方和数字数目
            # dp[j] = min(dp[j], dp[j-i]+1)
            coins = []
            for i in range(1, n):
                if i**2 <= n:
                    coins.append(i**2)
            
            dp = [float('inf')] * (n+1)
            dp[0] = 0
            dp[1] = 1
    
            # 先遍历物体，组合
            for i in range(len(coins)):
                # 直接通过这里做出限制，然后就可以避免多余的循环和判定
                for j in range(i**2, n+1):
                    # print(coins[i])
                    if dp[j-coins[i]] != float('inf'):
                        dp[j] = min(dp[j], dp[j-coins[i]]+1)
    
            # 先遍历背包(排列)
            for j in range(1, n+1):
                for i in range(len(coins)):
                     if dp[j-coins[i]] != float('inf'):
                        dp[j] = min(dp[j], dp[j-coins[i]]+1)
            
            return dp[n] 
    ```

    

19. 139单词拆分

    ```python
    class Solution:
        def wordBreak(self, s: str, wordDict: List[str]) -> bool:
            # dp[j]就是s长度到j的时候是否可以由字典组成
            # dp[j] = dp[j] or dp[j-i]
    
            dp = [False] * (len(s)+1)
            dp[0] = True
            
            # 这个问题其实是排列，因为要求了word的顺序
            for j in range(1, len(s)+1):
                for word in wordDict:
                    # 以文字长度去遍历， 那么过往的单词能否组成，并且右开
                    # print(s[j-len(word):j])
                    if j >= len(word):
                        dp[j] = dp[j] or (dp[j-len(word)] and word == s[j-len(word):j]) 
            
            return dp[len(s)]
    ```

    

20. 198 打家劫舍 [198. 打家劫舍 - 力扣（LeetCode）](https://leetcode.cn/problems/house-robber/)

    ```python
    class Solution:
        def rob(self, nums: List[int]) -> int:
            # dp[i]就是偷了第i家的和之前的最佳金额
            # dp[i] = max(dp[i-2] + nums[i], dp[i-1])
            # dp[0] = nums[0], dp[1] = max(nums[0], nums[1]), dp长度就是len(nums)
            if len(nums) == 1:
                return nums[0]
                
            dp = [0] * len(nums)
            dp[0] = nums[0]
            dp[1] = max(nums[0], nums[1])
    
            for i in range(2, len(nums)):
                dp[i] = max(dp[i-2] + nums[i], dp[i-1])
            return dp[-1]
    ```

    

21. 213打家劫舍 II  [213. 打家劫舍 II - 力扣（LeetCode）](https://leetcode.cn/problems/house-robber-ii/)
    这道题就是典型的结合具体问题分类讨论即可

    ```python
    class Solution:
        def rob(self, nums: List[int]) -> int:
            # 房子连在一起之后，能偷的最大数目是len(nums)//2
            # 最简单的就是算两种情况，分类讨论
    
            if len(nums) == 1:
                return nums[0]
            if len(nums) <= 2:
                return max(nums[0],nums[1])
    
            # 偷第一个，把最后一个扔掉
            nums1= nums[:len(nums)-1]
            dp = [0] *(len(nums1))
            dp[0] = nums1[0]
            dp[1] = max(nums1[0], nums1[1])
    
            for i in range(2, len(nums1)):
                dp[i] = max(dp[i-1], dp[i-2]+nums1[i])
    
            res1 = dp[len(nums1)-1]
            
            # 第一家不偷， 扔掉第一个
            nums1= nums[1:]
            dp[0] = nums1[0]
            dp[1] = max(nums1[0], nums1[1])
            for i in range(2, len(nums1)):
                dp[i] = max(dp[i-1], dp[i-2]+nums1[i])
            res2 = dp[len(nums1)-1]
            
            
            return max(res1, res2)
    ```

    

22. 337 打家劫舍 [337. 打家劫舍 III - 力扣（LeetCode）](https://leetcode.cn/problems/house-robber-iii/)

    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def rob(self, root: Optional[TreeNode]) -> int:
            # dp= [,]两个元素分别代表不偷该节点得到最大金额，偷该节点的最大金额
    
            def traversal(node):
                if not node:
                    # 递归终止条件，就是遇到了空节点，那肯定是不偷的
                    return (0, 0)
                
                left = traversal(node.left)
                right = traversal(node.right)
    
                # 不偷当前节点，然后偷子节点，那么需要判断左右子节点的最大值
                val_0 = max(left[0], left[1]) + max(right[0],  right[1])
    
                # 偷当前节点，那么子节点就不能偷
                val_1 = node.val + left[0] + right[0]
    
                return (val_0, val_1)
            
            res = traversal(root)
            return max(res)
    ```

23. 121买卖股票的最佳时机
    **从这里开始其实包含两种状况下的dp，dp还是一维dp，但是后面第二维度是二分类的结果**

    -  只能买卖一次，要引入持有的概念,否则卖出之后的状态无法表示
    - 今天持有股票要和-prices[i]比较，因为需要扣费

    ```python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            # 贪心
            low = float('inf')
            res= 0
            for i in range(len(prices)):
                low = min(low, prices[i])
                res = max(res, prices[i]-low)
            return res
        
            # 同样的是一维dp加上一个判断，然后是两种情况下的dp递推公式
            # 今天不持有股票所得的最大现金， 那么就是昨天不持有，或者是昨天买了今天卖出了
            # dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]) 
            # 今天持有股票所得最大现金，那么就是昨天就持有，和-prices比较
            # dp[i][1] = max(dp[i-1][1], -prices[i])
    
            dp = [[0]*2 for _ in range(len(prices))]
            dp[0][0] = 0
            dp[0][1] = -prices[0]
    
            for i in range(1, len(prices)):
                dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
                dp[i][1] = max(dp[i-1][1], -prices[i])
            
            return dp[len(prices)-1][0]
    ```

24. 122 买卖股票的最佳时机 II [122. 买卖股票的最佳时机 II - 力扣（LeetCode）](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

    ```python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            # 这次是可以买卖多次，但是手上最多一只股票
            # 今天不持有，昨天也没买或者昨天买今天卖
            # dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
            # 今天持有：昨天没买，今天买了或者昨天就买了
            # dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
            dp = [[0]*2 for _ in range(len(prices))]
            dp[0][1] = -prices[0]
    
            for i in range(1, len(prices)):
                dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
                dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
    
            return dp[len(prices)-1][0]
    ```

    

25. 123 买卖股票的最佳时机 III

    ```python
    ```

    

26. IV

27. 买卖股票的最佳时机 冷冻期 [309. 买卖股票的最佳时机含冷冻期 - 力扣（LeetCode）](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

    **之前的问题是只有两种状态的判断，现在是有多种情况判断，而且每种情况的状态转移不同**

    ```python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            # 之前的问题是只有两种状态的判断，现在是有多种情况判断，而且每种情况的状态转移不同
            # 分成持有股票、卖出、冻结、保持卖出四种状态
            # dp[i][0] = max(dp[i-1][0], dp[i-1][2]-prices[i] ,dp[i-1][3]-prices[i]) # 今天持有，只能昨天就持有或者是今天买的，而今天买只能昨天是持续卖出状态或者冻结期
            # dp[i][1] = dp[i-1][0] + prices[i] #今天能卖，昨天一定是持股的
            # dp[i][2] = dp[i][1] # 今天冻结，昨天得卖
            # dp[i][3] = max(dp[i][3], dp[i][2]) # 昨天冻结或者是昨天也是卖出状态
    
            dp = [[0]*4 for _ in range(len(prices))]
            dp[0][0] = -prices[0]
    
            for i in range(1, len(prices)):
                dp[i][0] = max(dp[i-1][0], dp[i-1][2]-prices[i] ,dp[i-1][3]-prices[i])
                dp[i][1] = dp[i-1][0] + prices[i]
                dp[i][2] = dp[i-1][1]
                dp[i][3] = max(dp[i-1][3], dp[i-1][2])
                # print(dp)
            
            return max(dp[-1][1], dp[-1][2], dp[-1][3]) # 对所有不持股票的条件做判断
    ```

28. 买卖股票的最佳时机 手续费 [714. 买卖股票的最佳时机含手续费 - 力扣（LeetCode）](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)
    和多次买卖股票，但是仅能持有一支是一样的解法，但是这道题在卖出时需要支付交易费

    ```python
    class Solution:
        def maxProfit(self, prices: List[int], fee: int) -> int:
            # dp[i][]代表持有还是不持有
            # 今天不持有，只有昨天不持有或者昨天有今天卖了
            # dp[i][0] = max(dp[i-1][0], dp[i-1][1] +prices[i]-fee[i])
            # 今天持有，只有昨天持有或者今天买了
            # dp[i][1] = max(dp[i-1][1], dp[i-1][0] -prices[i]-fee[i])
            dp = [[0]*2 for _ in range(len(prices))]
            dp[0][1] = dp[0][1] -prices[0]
    
            for i in range(1, len(prices)):
                dp[i][0] = max(dp[i-1][0], dp[i-1][1] +prices[i]-fee)
                dp[i][1] = max(dp[i-1][1], dp[i-1][0] -prices[i])
    
            return dp[-1][0]
    ```

29. 最长递增子序列

     子序列问题是动态规划解决的经典问题

    ```python
    class Solution:
        def lengthOfLIS(self, nums: List[int]) -> int:
            # dp[i]代表了以nums[i]结尾的最长递增子序列长度
            # 递归最大的思想就是遍历的同时保存结果，然后后面的计算在先前的计算基础上完成，从而减少计算重复
            # 所以递归最重要的是从小的元素思考，然后想如何使用这些结果进行状态转移
            # dp[i] = max(dp[i], dp[j]+1) ，如果nums[j] < nums[i]，那么加1
            # dp[i] = 1， 初始化1，因为自身保证了长度下限
    
            dp = [1] * (len(nums))
            for i in range(len(nums)):
                # 这里类似于双指针，然后内部不断与外部结果比较
                for j in range(0, i):
                    # 严格递增
                    if nums[j] < nums[i]:
                        dp[i] = max(dp[j]+1, dp[i])
            
            # print(dp)
            # 最后一个元素不一定是最大值，例如[1,3,6, 90, 7] 所以找最大值
            res= 0 
            for ele in dp:
                res = max(res, ele)
            return res
    ```

30. 674最长连续递增子序列
    贪心和动规都可以，但是动规会超时一些算例

    ```python
    class Solution:
        def findLengthOfLCIS(self, nums: List[int]) -> int:
            # 贪心
            # 只要不满足的就重新计算
            cur = 0 
            res = 1
            length = 1
            for fast in range(1, len(nums)):
                if nums[fast] > nums[fast-1]:
                    length += 1
                    res = max(res, length)
                else:  
                    length = 1
    
            return res  
            
            #动态
            #这题与上一题的唯一区别在于，这道题要求连续，所以比较的对象变成了j前后两个
            # 下面这种想法并不准确，因为后者的条件包含前者
            # if nums[j]<nums[i] and nums[j]<nums[j+1]
    
            # 直接去掉前面的比较，只比较后面的if nums[j]<nums[j+1]:
    
            # res= 1
            # dp = [1] * (len(nums))
            # for i in range(len(nums)):
            #     for j in range(i):
            #         if nums[j]<nums[j+1]:
            #             dp[i] = max(dp[i], dp[j]+1)
            #             res = max(res, dp[i])
            #         else:
            #             dp[i] = 1
    
            # return res
    ```

    

31. 最长重复子数组 [718. 最长重复子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/)

    二维dp数组一定要记得和内外部循环列表的长度一致

    ```python
    class Solution:
        def findLength(self, nums1: List[int], nums2: List[int]) -> int:
            # 使用二维dp数组
            # dp[j][j]表示在第一个数组nums1[i]和第二个数组nums1[j]结尾的的最大重复序列
            # 并非求递增数组
    
            res = 0
            dp = [[0] * (len(nums2)+1) for _ in range(len(nums1)+1)]
            # 这里的行和列需要和nums1和nums2对应一致
            for i in range(1,len(nums1)+1):
                for j in range(1,len(nums2)+1):
                    if nums1[i-1] == nums2[j-1]:
                        dp[i][j] = dp[i-1][j-1]+1
                        res = max(res, dp[i][j])
            return res
    ```

32. 1143 最长公共子序列 / 1035不想交的线

    ```python
    class Solution:
        def longestCommonSubsequence(self, text1: str, text2: str) -> int:
            # 递推公式是一样的，但是这里面是可以离散的选择也就是
            # 状态方程， 这次的状态转移是潜在的，多种方式的
            # dp[i][j]可以通过dp[i-1][j-1], dp[i][j-1], dp[i-1][j]得到
    
            # 在需要对前后值进行比较的时候，就需要额外设置一个长度len()+1
            dp = [[0] * (len(text2)+1) for _ in range(len(text1)+1)]
    
            for i in range(1, len(text1)+1):
                for j in range(1, len(text2)+1):
                    if text1[i-1] == text2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1 
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            # print(dp)
            return dp[len(text1)][len(text2)]
    ```

    

33. 53 最大子数组和 [53. 最大子数组和 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-subarray/)
    同样的，需要做一个额外的比较

    ```python
    class Solution:
        def maxSubArray(self, nums: List[int]) -> int:
            # dp[i] 代表前i个元素的最大子序列和
            # 在遍历到当前元素的时候，如果前一个元素的最大子数组和小于0，那么等于自己，否则加上
    
            if len(nums) <= 1:
                return nums[0]
    
            dp = [0] * (len(nums) + 1)
            dp[1] = nums[0]
            for i in range(1, len(nums)+1):
                if dp[i-1] > 0:
                    dp[i] = dp[i-1] + nums[i-1]
                else:
                    dp[i] = nums[i-1]
            
            return max(dp[1:])
    ```

34. 392 判断子序列
    然后到了**编辑距离**的环节，这里面需要额外对结果做一个判断

    ```python
    class Solution:
        def isSubsequence(self, s: str, t: str) -> bool:
            # 和最长公共子序列类似，这里的心意在与最后判断公共最大长度与s是否相等
    
            dp = [[0]*(len(s)+1) for _ in range(len(t)+1)]
    
            for i in range(1, len(t)+1):
                for j in range(1, len(s)+1):
                    if s[j-1] == t[i-1]:
                        dp[i][j] = dp[i-1][j-1] + 1 
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
            # print(dp)
            return True if dp[-1][-1] == len(s) else False
    ```

    35. 115 不同的子序列 [115. 不同的子序列 - 力扣（LeetCode）](https://leetcode.cn/problems/distinct-subsequences/) (没懂！)

        ```python
        class Solution:
            def numDistinct(self, s: str, t: str) -> int:
                # 显然有用二维dp数组，然后dp[i][j]下标是以i-1,j-1结尾的个数
                dp = [[0]*(len(t)+1) for _ in range(len(s)+1)]
        
                for i in range(len(s)):
                    dp[i][0] = 1
                for j in range(1, len(t)):
                    dp[0][j] = 0
                
                for i in range(1, len(s)+1):
                    for j in range(1, len(t)+1):
                        if s[i-1] == t[j-1]:
                            # 如果s去掉一个元素元也能够得到这个结果，这里其实涉及到删除操作
                            dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
                        else:
                            dp[i][j] = dp[i-1][j]
                
                return dp[-1][-1]
        ```

        

    36. 583 两个字符串的删除操作 

        ```python
        class Solution:
            def minDistance(self, word1: str, word2: str) -> int:
                dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
        
                # 初始化，删除对齐
                for i in range(len(word1)+1):
                    dp[i][0] = i 
                for j in range(len(word2)+1):
                    dp[0][j] = j
                
                for i in range(1, len(word1)+1):
                    for j in range(1, len(word2)+1):
                        if word1[i-1] == word2[j-1]:
                            # 如果相等的话，不用执行操作
                            dp[i][j] = dp[i-1][j-1]
                        else:
                            # dp[i][j-1]这里其实就是删除操作，删除word[0:j]的最后一个元素
                            dp[i][j] = min(dp[i][j-1]+1, dp[i-1][j]+1, dp[i-1][j-1]+2)
                
                # print(dp)
                return dp[-1][-1]
        ```

        

37. 72 **编辑距离**   [72. 编辑距离 - 力扣（LeetCode）](https://leetcode.cn/problems/edit-distance/)

    ```python
    class Solution:
        def minDistance(self, word1: str, word2: str) -> int:
    
            # 能做的操作里面：增删替，其中增加word1某个字母和删掉word2多的某个字母是一样的操作数
            # 增 = 删
            # 而对于替换，其实就是在word1[i-1] != word2[j-1]的时候，将不相等的元素更换，这里操作数加1，
            # 所以这里很抽象，不需要实际地去换，而是判断是否进行操作，操作数怎么处理
    
            dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
    
            # 初始化，删除对齐
            for i in range(len(word1)+1):
                dp[i][0] = i 
            for j in range(len(word2)+1):
                dp[0][j] = j
            
            for i in range(1, len(word1)+1):
                for j in range(1, len(word2)+1):
                    if word1[i-1] == word2[j-1]:
                        # 如果相等的话，不用执行操作
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        # dp[i][j-1]这里其实就是删除操作，删除word[0:j]的最后一个元素
                        dp[i][j] = min(dp[i][j-1]+1, dp[i-1][j]+1, dp[i-1][j-1]+1)
            
            # print(dp)
            return dp[-1][-1] 
    ```

    

38. 647回文子串
    很有意思的一道题目

    ```python
    class Solution:
        def countSubstrings(self, s: str) -> int:
            # dp[i]的定义是前i个元素的回文子串,但是不对，因为递推关系难以建立
            
            # 使用二维dp[i][j]分别代表了s[i:j]是不是一个回文子串，
            # 首先需要判断s[i]和s[j]是不是相等的，如果相等：
            # 在i=j的时候，指向的是一个字母，j-i=1的时候,也加1
            # j-i > 1的时候，向内去判断内部是不是回文的，内部回文dp[i+1][j-1]两边相等，那么加1
    
            dp = [[False]*(len(s)+1) for _ in range(len(s)+1)]
            res = 0
            for i in range(len(s)+1, 0, -1):
                for j in range(i, len(s)+1):
                    if s[i-1] == s[j-1]:
                        if j-i<=1:
                            dp[i][j] = True
                            res += 1
                        else:
                            if dp[i+1][j-1]:
                                dp[i][j] = True
                                res += 1
            
            # print(dp)
            return res
    ```

    

39. 516 最长回文子序列  [516. 最长回文子序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-palindromic-subsequence/)

    ```python
    class Solution:
        def longestPalindromeSubseq(self, s: str) -> int:        
            # 使用二维dp[i][j]分别代表了s[i:j]是不是一个回文子串，
            # 如果s[i-1] == s[j-1]，那么dp[i][j] = dp[i+1][j-1]+2,因为两边长度增加了
            # 在i=j的时候，指向的是一个字母，j-i=1的时候,也加1
            # j-i > 1的时候，向内去判断内部是不是回文的，内部回文dp[i+1][j-1]两边相等，那么加1
    
            dp = [[0]*(len(s)+1) for _ in range(len(s)+1)]
            for i in range(1, len(s)+1):
               dp[i][i] = 1 
    
            for i in range(len(s)+1, 0, -1):
                # 这里不考虑相等的情况
                for j in range(i+1, len(s)+1):
                    if s[i-1] == s[j-1]:
                        dp[i][j] = dp[i+1][j-1]+2
                    else:
                        dp[i][j] = max(dp[i][j-1], dp[i+1][j])
            
            # print(dp)
            return dp[1][-1]
    ```

    
