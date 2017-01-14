  

## 前言

本人最近正在学习深度学习以及tensorflow，在此记录一些学习过程中看到的有价值的参考资料，并且写下一点我自己的初步理解。

* 环境

  win10 64+anaconda3(python3.5)+tensorflow0.12.1

  关于windows下CUDA等配置，请参考下文：

  [windows 10 64bit下安装Tensorflow+Keras+VS2015+CUDA8.0 GPU加速](http://www.jianshu.com/p/c245d46d43f0)

* 部分参考/推荐资料

  [tensorflow官方引导文档](https://www.tensorflow.org/tutorials/recurrent/)

  [Udacity深度学习(即谷歌深度学习公开课)](https://classroom.udacity.com/courses/ud730/lessons/6370362152/concepts/63798118260923)

  [网友multiangle深度学习笔记](http://blog.csdn.net/u014595019/article/details/52677412)

  [运用TensorFlow处理简单的NLP问题](http://sharkdtu.com/posts/nn-nlp.html)

* 笔记目录
  * [TF学习笔记(一)-基础模型]()
  * [TF文档笔记(二)-PTBModel初步解析]()


## 二、Tutorial

​	tensorflow可以非常方便的构建**抽象模型**，在TF中**构建模型**和**训练**是完全分离开来的。TF通过以下几个概念完成了模型和训练的分离：**Tensor、Variable、placeholder以及Session**，

​	下面的资料部分参考[http://blog.csdn.net/u014595019/article/details/52677412]

*   Tensor

    tensor基本可以视作矩阵处理，如下面的代码就构造了一个1x2的0矩阵。

    ```python
    import tensorflow as tf # 在下面所有代码中，都去掉了这一行，默认已经导入
    a = tf.zeros(shape=[1,2])
    ```

*   Variable

        Variable表示变量，下面的代码就用最简单的方式构建了一个Variable。


    ​```python
          W = tf.Variable(tf.zeros(shape=[1,2]))
   ```


    与Tensor不同，Variable必须初始化以后才能使用，

 ```python
    tensor = tf.zeros(shape=[1,2])
    variable = tf.Variable(tensor)
    sess = tf.InteractiveSession()
    # print(sess.run(variable))  # 会报错
    sess.run(tf.initialize_all_variables()) # 对variable进行初始化
    print(sess.run(variable))
   ```

*   placeholder

    个人认为placeholder与Variable较为相似，区别在于placeholder常用来表示输入及输出，而Variable常用来表示中间变量。placeholder至少要求指定变量类型和shape。

    ```python
    x = tf.placeholder(tf.float32,[1, 28*28])
    y = tf.placeholder(tf.float32,[None, 10])
    ```
    上面x中[1,28*28]表示输入的数据有1行，每行有28x28个值。
     而y中出现了**[None,10]**这种语法，表示接受**任意行数**的数据，输出为10个值。

*   Session

        上面构建的Tensor、Variable、placeholder都属于抽象变量，需要通过Session控制模型运行(run)。

       Session非常重要，我在学习之初一直没有找到代码到底在哪里对model进行调用并更新值，直到后来偶然间发现通过session.run(model.something)就可以更新该目标及所有该目标所涉及到的值。假设有以下代码：

    ```python
     class Model(object):
        def __init__(self, is_training, config, input_):
            self._input = input_

            batch_size = input_.batch_size
            num_steps = input_.num_steps
            size = config.hidden_size
            vocab_size = config.vocab_size
            
    		# 一些相关设置
            
            output = tf.reshape(tf.concat(1, outputs), [-1, size])
            softmax_w = tf.get_variable(
                "softmax_w", [size, vocab_size], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
            logits = tf.matmul(output, softmax_w) + softmax_b
            loss = tf.nn.seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(input_.targets, [-1])],
                [tf.ones([batch_size * num_steps], dtype=data_type())])
            self._cost = cost = tf.reduce_sum(loss) / batch_size
    ```

         上面的代码可以构建一个抽象模型，但是他自己是无法主动运行的。必须在外部通过session.run()对其进行调用：

    ```python
              cost=session.run(model._cost, feed_dict={})
    ```

         这个run要求更新_cost，而cost会涉及loss，loss又要用到logits等许多参数，所以model就会回溯更新所有用到的值，并且在最后将求得cost返回。

         每一次run都会更新相当多的参数，那么如果我想获得多个参数数值应该怎么办？我在一开始就犯了在一个循环中用run获得多个参数值的错误，这样就会导致参数重复更新。正确的解决方法是：

    ```python
              fetches = {
                  "cost": model.cost,
                  "final_state": model.final_state,
                  "input_data":model._input_data,
                  "targets":model._targets,
              }
              vals = session.run(fetches, feed_dict)

              cost = vals["cost"]
              state = vals["final_state"]
    ```

         通过这种方式，就可以在一个run中获取多个参数的值。

         注意到有一个**feed_dict**变量，这个就是要提供给模型的input，留意一下稍后介绍。

*   mnist代码示例

    > 本小节会涉及到一些softmax、交叉熵、梯度下降等知识，如有需要个人推荐[https://classroom.udacity.com/courses/ud730/lessons/6370362152/concepts/63798118150923#]结合书籍和网上资料的方式补充一下这些必要内容。

    官方用mnist手写识别作为代码示例：[MNIST For ML Beginners](https://www.tensorflow.org/versions/r0.11/tutorials/mnist/beginners/)

    mnist中每个图片均为28x28,共有A-J10个分类。

    下面来演示如何用tf构造一个最简单的LogisticRegression Classifier，基本公式为：

    ​									$t=Wx+b$

    ​									$a=softmax(t)$

    * 构建抽象模型

      ```python
      # 建立抽象模型
      x = tf.placeholder(tf.float32, [None, 784]) # 输入占位符
      y = tf.placeholder(tf.float32, [None, 10])  # 输出占位符（预期输出）

      W = tf.Variable(tf.zeros([784, 10]))        
      b = tf.Variable(tf.zeros([10]))
      a = tf.nn.softmax(tf.matmul(x, W) + b)      # a表示模型的实际预测输出

      # 定义损失函数和训练方法
      cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1])) # 损失函数为交叉熵
      optimizer = tf.train.GradientDescentOptimizer(0.5) # 梯度下降法，学习速率为0.5
      train = optimizer.minimize(cross_entropy)  # 训练目标：最小化损失函数

      # 判断a和y是否匹配，并计算accuracy
      correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      ```

      * x为输入28*28特征值，y为输出的10种分类。
      * w和b要将输入的28*28转换为10种分类，利用softmax可以得到模型对某个图片应该划分哪个各类别的概率。
      * loss与交叉熵属于套路，掌握即可。
      * 最后要判断本次预测的精度，tf.argmax(a,1)可以找到各行a的最大值的index，将其与对应的y的index比较，得到一个Ture or False矩阵，代表本轮预测的正确与否，最后计算true的比例求得accuracy。

    * 开始训练

      ```python
      from tensorflow.examples.tutorials.mnist import input_data

      flags = tf.app.flags
      FLAGS = flags.FLAGS
      flags.DEFINE_string('data_dir', r'C:\Users\hasee\Desktop\tempdata', 'Directory for storing data') # data_dir表示数据存放路径

      mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)   # 读取数据集

      session=tf.InteractiveSession()
      tf.initialize_all_variables().run()

      # fetches在session要run多个值时会用到
      fetches={
          'step':train_step,
          'intermediate_accuracy':accuracy
      }
      begin_time=time()
      for i in range(1000):
          batch_xs, batch_ys = mnist.train.next_batch(1000)    # 获得一批100个数据
          train_step.run({x: batch_xs, y: batch_ys})   # 给训练模型提供输入和输出
          # session.run(train_step, {x: batch_xs, y: batch_ys}) # 和上面这句是等效的

          # 如果想要把模型的中间结果输出看看，使用方法一。
          # 方法一：fetches为想要查看的值，已经在外部定义。此方法在我的机器上耗时7.5s
          # vals=session.run(fetches, {x: batch_xs, y: batch_ys}) # 和上面这句是等效的
          # intermediate_accuracy=vals['intermediate_accuracy']

          # 方法二：分别run各值。这种方法在我的机器上耗时35s，而且在很多情况下会导致model不能正常运行(我学习时遇到的大坑之一)。
          # session.run(train_step, {x: batch_xs, y: batch_ys}) # 和上面这句是等效的
          # ans=session.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
      print(session.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
      print(time()-begin_time)

      ```
       在model外部只需要为model提供数据并通过session让model运作起来就可以了，模型会在运行时按照预设的代码完成求导、计算loss等操作并更新内部数值。最后输出accuracy，约为0.92.

    * 完整代码：

      强烈建议和我一样的初学者尽量透彻地理解这份代码的意义，然后将其裸写一遍。之后无论是CNN还是LSTM都逃不开这种最基础的建立模型、训练模型的套路。

      ```python
        # encoding:utf-8

        import tensorflow as tf

        # 建立模型
        x=tf.placeholder(tf.float32, [None, 28 * 28])
        y=tf.placeholder(tf.float32, [None, 10])

        w=tf.Variable(tf.zeros([28*28,10]))
        b=tf.Variable(tf.zeros([10]))
        a=tf.nn.softmax(tf.matmul(x, w) + b)

        cross_entropy=tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))
        optimizer=tf.train.GradientDescentOptimizer(0.5)
        train_step=optimizer.minimize(cross_entropy)

        correct_prediction=tf.equal(tf.argmax(a,1), tf.argmax(y, 1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        # 输入数据，调用模型

        from tensorflow.examples.tutorials.mnist import input_data
        from time import time

        flags = tf.app.flags
        FLAGS = flags.FLAGS
        flags.DEFINE_string('data_dir', r'C:\Users\hasee\Desktop\tempdata', 'Directory for storing data') # 把数据放在/tmp/data文件夹中

        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)   # 读取数据集

        session=tf.InteractiveSession()
        tf.initialize_all_variables().run()

        fetches={
            'step':train_step,
            'intermediate_accuracy':accuracy
        }
        begin_time=time()
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(1000)    # 获得一批100个数据
            train_step.run({x: batch_xs, y: batch_ys})   # 给训练模型提供输入和输出
            # session.run(train_step, {x: batch_xs, y: batch_ys}) # 和上面这句是等效的

            # 如果想要把模型的中间结果输出看看，使用方法一。
            # 方法一：fetches为想要查看的值，已经在外部定义。此方法在我的机器上耗时7.5s
            # vals=session.run(fetches, {x: batch_xs, y: batch_ys}) # 和上面这句是等效的
            # intermediate_accuracy=vals['intermediate_accuracy']

            # 方法二：分别run各值。这种方法在我的机器上耗时35s，而且在很多情况下会导致model不能正常运行(我学习时遇到的大坑之一)。
            # session.run(train_step, {x: batch_xs, y: batch_ys}) # 和上面这句是等效的
            # ans=session.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print(session.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
        print(time()-begin_time)
      ```


