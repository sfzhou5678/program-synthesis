# 简化版FlashExtract实现步骤与核心思路

> ### **假设：**
>
> 1. 待提取的数据全都是半结构化text数据
> 2. 待提取的数据全都是单行，不存在多行现象(极大简化问题)

### 一、实现步骤

1. #### 根据positiveExample数量确定使用SeqRegion还是普通Region

   * 如果只有一个example,那么转化为最普通的FlashFill问题
   * 如果有多个examples，那么进入步骤2

2. #### 处理多个examples的SeqRegion

   大致思想为: 

   * 先提取父Region中的有效域S(比如利用filter找到target的所在行，详见步骤3)
   * 然后在有效域中利用FlashFill查找合适的substr方法F(LineMap等，详见步骤4)
   * 最后将F和S合并并且cleanUp即得到解

3. #### 提取父Region中的有效域S(FlashExtract框架最为核心的因素)

   1. 对每一个examples[i]，首先找到target[i]的所在行，并且将newTarget[i]设为target[i]所在行整行内容

   2. ★★然后利用方法2.1找到所有newTarget[i]的方法(**即找出此行的特征**),记作witness[i] (重要: witness[i]是一个方法集合)

   3. 但是这些witness不一定通用，所以依次将witness[ i ]-[ j ] 作用于Region从而得到Region中的某一行Lij,再将Lij和期望输出Yi一起送到方法4的FlashFill中搜寻结果

   4. (在S.learn()中)merge所有witness,找到一个通用的FilterBool方法, 利用此方法提取的SeqRegion即为有效域S

      例：

      LS = FilterInt(0,1,FilterBool(λx : EndsWith([Number, Quote],x),split(R 0 ,‘\n’)))

      S=LS.interpret(Region)

      PS: 其中"split(R 0 ,‘\n’)" 就是指splitByLine

   5. ★★隐式negativeExamples: 所有positiveExample之前未选中的line均视为negativeExample(在数据全都是单行的情况下)。

4. #### 在有效域S中利用FlashFill查找合适的substr方法F

   已经提取得到S(S是一个lineList, )

   ​

### 二、核心思想

1. 提取dynamicToken的方法

   比较几个examples截断处(选出区域)前后的数据，找到共同字串作为dynamicTok，如：

   <tr><td>**Russell Smith**</td><td>Russell.Smith@contoso.com</td><td>London</td></tr>
   <tr><td>**David Jones**</td><td>David.Jones@contoso.com</td><td>Manchester</td></tr>

   提取出Russell Smith和David Jones，那么两个reion的前共同字串为<tr><td>，后共同字串为</td><td>，可以把这两个当作dynamicTok

   PS: 那能不能考虑直接提取两个region之间相似的区域作为dynamicTok？

2. 提取行特征的方法

   定位某个line的方法有: 

   | {start,end}with(r); | contains(r,k)

   | pred{start,end} with(r) ; | predContains(r,k)

   | succ{start,end} with(r) ;  | succContains(r,k)

   其中r可以由多个token拼接而成，暂定最大拼接深度为3. 暂定深度优先级为2>3>1。

    **PS: 暂时只考虑{start,end} with(r)** 

   首先调用buildMatch用所有token对input做一次匹配，

   然后在正向(start with)处理时先查找所有index=0的match, 然后根据DFS依次深入直到depth>3.

   反向(end with)处理时同理。

   上述操作可以得到一组提取line的解，然后通过positive和negative两种Examples进行筛选。

   > **positiveExamples**由用户输入，**negativeExamples**为positiveExamples之前的所有行。
   >
   > 上面求得的exp必须满足positive和negative两种examples。

1. xxx

