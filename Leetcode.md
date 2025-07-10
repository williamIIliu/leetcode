## 贪心

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

