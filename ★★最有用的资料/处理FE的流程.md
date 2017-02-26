# FlashExtract核心问题及解决方案

### 一、操作流程概述

> **PS: 要求输入为半结构化text数据。**

1. (可略过)手动框选Element Region(可能是多行)，要求系统自动识别并标注全文所有的Element。
2. 框选目标结构化参数，要求能自动识别出提取范围(1. ancestor 2. 所在region中的哪一行(一个region中会有多行数据，要求找出目标参数所在行的特征))，并且提取出所有同级参数。



### 二、核心问题及解决方案

1. 如何确定Region的边界？

   > 解决方案：
   >
   > (1) 如果Region是单行的，问题转化为问题2
   >
   > (2) 如果Region是多行的，问题转化为处理行元素的方法，可以再分解为多个问题2。只要找到Example Region的首行(开头)及末行(结尾)**特征**，之后再全文中找到所有subStr(p1,p2)即可。
   >
   > 如：
   >
   > a. Extract the strings **starting at** each line **starting with** Left Bracket, and **ending at** 
   > the last end of Dot (references)
   >
   > b. Extract the strings **starting at** each line **starting with** Left-than◦Alphanumeric, and **ending at** 
   > the first occurrence of Line Separator after Line Separator◦"                </div>"  (multi lines html data)**(可以用RMM改进？只要从文章末尾依次遍历n行，每次都去匹配上面取出来的模式，找到那种刚好首尾各匹配一次的区域即为目标区域，然后从这里开始继续执行RMM。执行完毕之后判断提取结果与examples是否一致。)**
   >
   > ​
   >
   > ​

2. 如何识别出所在行的**特征**？

   有时没有为半结构化数据划分region就直接提取数据，那么此时只能通过自行判断所在行的特征(并找出所有同级行)来处理问题。

   要求：

   (1) 正确确定行开头的标志(若数据是单行的，则标志是beginning of the line)，如从“<tr><td>Russell Smith</td><td>Russell.Smith@contoso.com</td><td>London</td></tr>”中识别出” starting with "<tr><td>"◦Words/dots/hyphens◦WhiteSpace” (constr+reg) 或从”[7] S. Gulwani. Automating string processing... “中识别出” **starting with** Left Bracket" (reg)

   (2) 正确确定结尾的标志(若数据是单行的，则标志是end of the line)，如从”[7] S. Gulwani. Automating string processing... “中识别出"**ending at** the last end of Dot "(reg),或从“html多行数据”中识别出“ending at the first occurrence of Line Separator after Line Separator◦"                </div>" (reg+constr)

   > 解决方案：
   >
   > 方法等同于FlashFill，还是用constr+reg的方式识别出左右两边最有特点的pos。
   >
   > 如：
   >
   > a. Extract all lines **starting with** "<tr><td>"◦Words/dots/hyphens◦WhiteSpace (to the end) (single line semi-structed data)
   >
   > b. From all lines (from the head and )**ending with** Left-than◦Words/dots/hyphens◦Greater-than (single line html data ending with <br>)
   >
   > ​




### 三、问题总结

FlashExtract最大的难点在于如何提取与处理Region，成功提取Region之后，问题就转换成了FlashFill。

提取Region的方法其实仍然与FF类似，只不过它是与自己进行验证。比如FF中的a 1,b 2,c 3,d 4 -> 1 2 3 4，positive examples=1和2，做的其实就是一个Loop(num)，如果将num替换为一个复杂的结构，那么这个问题就变成了region提取。

Region最重要的一点是自己对解进行验证，具体验证方法待定。

以1,2,3,4为例：

方法1: 提取出所有alpha+空格与逗号之间的数据，那么RMM时





1. **提取行开头结尾(即region)**与单行数据中**提取pos**的区别？

   region需要考虑上下文的关联(比如reference的例子)，可能可以用RMM进行改进；pos不考虑上下文，只考虑提取此位置的最优解。

2. 解如何排序

3. ​



### 四、程序思路与流程

1. 如何判断要提取的是seqRegion还是普通Region？

   A: 在一个父region中，同种label仅出现1次则认为是普通Region提取，同种label出现多次则认为是seqRegion提取。

2. ​

