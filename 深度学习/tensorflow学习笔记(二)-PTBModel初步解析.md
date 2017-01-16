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

    ​
* 笔记目录
  * [tensorflow学习笔记(一)-基础模型](http://blog.csdn.net/a343902152/article/details/54428985)
  * [tensorflow学习笔记(二)-PTBModel初步解析](http://blog.csdn.net/a343902152/article/details/54429096)

## 二、PTB

本人所用anaconda3,ptb位于Anaconda3\Lib\site-packages\tensorflow\models\rnn\ptb目录下，共包含

- ptb_word_lm.py
- reader.py

两个主要文件。其中reader是PTB模型处理数据的工具包。PTBModel、main都位于ptb_word_lm中。

和之前的Tutorial一样，PTB也是分为**构建抽象模型**和**训练**两大步骤。

官方文档位于：https://www.tensorflow.org/tutorials/recurrent



1. 配置说明

   这份官方代码非常有心的设置了4种不同大小的配置，分别为small，medium、large和test，以small为例：

   ```python
   class SmallConfig(object):
       """Small config."""
       init_scale = 0.1    # 相关参数的初始值为随机均匀分布，范围是[-init_scale,+init_scale]
       learning_rate = 1.0 # 学习速率，此值还会在模型学习过程中下降
       max_grad_norm = 5   # 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小
       num_layers = 2      # LSTM层数
       num_steps = 20      # 分隔句子的粒度大小，每次会把num_steps个单词划分为一句话(但是本模型与seq2seq模型不同，它仅仅是1对1模式，句子长度应该没有什么用处)。
       hidden_size = 200   # 隐层单元数目，每个词会表示成[hidden_size]大小的向量
       max_epoch = 4       # epoch<max_epoch时，lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
       max_max_epoch = 13  # 完整的文本要循环的次数
       keep_prob = 1.0     # dropout率，1.0为不丢弃
       lr_decay = 0.5      # 学习速率衰减指数
       batch_size = 20     # 和num_steps共同作用，要把原始训练数据划分为batch_size组，每组划分为n个长度为num_steps的句子。
       vocab_size = 10000  # 单词数量(这份训练数据中单词刚好10000种)
   ```

   另有以下配置，可以设置要选用的config(下面为small)、数据地址、输出存储地址等。

   ```python
   flags = tf.flags
   logging = tf.logging

   flags.DEFINE_string(
       "model", "small",
       "A type of model. Possible options are: small, medium, large.")
   flags.DEFINE_string("data_path", r'C:\Users\hasee\Desktop\tempdata\lstm\simple-examples\data', "data_path")
   flags.DEFINE_string("save_path", r'C:\Users\hasee\Desktop\tempdata\lstm\simple-examples\data\res',
                       "Model output directory.")
   flags.DEFINE_bool("use_fp16", False,
                     "Train using 16-bit floats instead of 32bit floats")

   FLAGS = flags.FLAGS
   ```


1. PTBModel

   在class PTBModel的init()中构建了一个抽象LSTM模型。

   * lstm_cell和initial_state

     ```python
     # Slightly better results can be obtained with forget gate biases
     # initialized to 1 but the hyperparameters of the model would need to be
     # different than reported in the paper.
     # 注释指的是如果将forget_bias=0.0改为1.0会得到更好的结果，但是这将与论文中的描述不符。
     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
     if is_training and config.keep_prob < 1:
         lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
             lstm_cell, output_keep_prob=config.keep_prob)
     cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

     self._initial_state = cell.zero_state(batch_size, data_type())
     ```

     使用BasicLSTMCell构建一个基础LSTM单元，然后根据keep_prob来为cell配置dropout。最后通过MultiRNNCell将num_layers个lstm_cell连接起来。

     在LSTM单元中，有2个**状态值**，分别是c和h。

     更多基础知识请见[tensorflow笔记：多层LSTM代码分析 ](http://blog.csdn.net/u014595019/article/details/52759104)

     * 问：为什么使用BasicLSTMCell而不是LSTMCell？

       答：根据[解读tensorflow之rnn ](http://blog.csdn.net/mydear_11000/article/details/52414342)，官方文档给出如下描述：

       ![img](http://ww3.sinaimg.cn/large/901f9a6fjw1f5vhejbskrj20o408wwhj.jpg)

       BasicLSTMCell没有实现clipping，projection layer，peep-hole等一些lstm的高级变种，仅作为一个基本的basicline结构存在，如果要使用这些高级variant要用LSTMCell这个类。

       由于我们现在只是想搭建一个基本的lstm-language model模型，现阶段BasicLSTMCell够用。这就是为什么这里用的是BasicLSTMCell这个类而不是别的什么。

   * embedding

     ```python
     with tf.device("/cpu:0"):
         embedding = tf.get_variable(
             "embedding", [vocab_size, size], dtype=data_type())
         # input_.input_data为外部输入的id形式的数据，通过embedding_lookup()将ids转换为词向量形式inputs。
         inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
     ```

     在这里embedding表示词向量矩阵。此矩阵共有vocab_size行(在这里为10000)，每一行都是一个hidden_size维向量，随着模型的训练，embedding内部权值会不断更新，最终可以得到各个词的向量表示。

     ​

   * outputs与loss

     这里与基础模型的套路大致一致：

     ```python
     outputs = []
     state = self._initial_state
     with tf.variable_scope("RNN"):
         for time_step in range(num_steps):
             if time_step > 0: tf.get_variable_scope().reuse_variables()
             # 这个cell(inputs[:, time_step, :], state)会调用tf.nn.rnn_cell.MultiRNNCell中的__CALL__()方法
             #  TODO __CALL__()的注释说：Run this multi-layer cell on inputs, starting from state.但是还没看该方法实际做了什么
             (cell_output, state) = cell(inputs[:, time_step, :], state)
             outputs.append(cell_output)
     # 下面套路和基础模型一致，y=wx+b
     # x=output,y=targets
     output = tf.reshape(tf.concat(1, outputs), [-1, size])
     softmax_w = tf.get_variable(
         "softmax_w", [size, vocab_size], dtype=data_type())
     softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
     logits = tf.matmul(output, softmax_w) + softmax_b
     self._logits=logits

     # 将loss理解为一种更复杂的交叉熵形式：与基础模型中的代码类似：
     # cross_entropy=tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))
     loss = tf.nn.seq2seq.sequence_loss_by_example(
                 [logits],
                 [tf.reshape(input_.targets, [-1])],
                 [tf.ones([batch_size * num_steps], dtype=data_type())])
     # 上述loss是所有batch上累加的loss，取平均值作为_cost
     self._cost = cost = tf.reduce_sum(loss) / batch_size
     self._final_state = state
     ```

   * lr与梯度下降

     参考[解读tensorflow之rnn ](http://blog.csdn.net/mydear_11000/article/details/52414342)

     在此lstm模型运行过程中需要动态的更新gradient值。

     官方文档说明了这种操作：

     ![img](http://ww3.sinaimg.cn/large/901f9a6fjw1f5vsocbai8j20or06pmzq.jpg)

     并给出了一个例子：

     ```python
     # Create an optimizer.
     opt = GradientDescentOptimizer(learning_rate=0.1)

     # Compute the gradients for a list of variables.
     grads_and_vars = opt.compute_gradients(loss, <list of variables>)

     # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
     # need to the 'gradient' part, for example cap them, etc.
     capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]

     # Ask the optimizer to apply the capped gradients.
     opt.apply_gradients(capped_grads_and_vars)
     ```

     模仿这个代码，我们可以写出如下的伪代码：

     ```python
     optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)

     # gradients: return A list of sum(dy/dx) for each x in xs.
     grads = optimizer.gradients(self._cost, <list of variables>)
     clipped_grads = tf.clip_by_global_norm(grads, config.max_grad_norm)

     # accept: List of (gradient, variable) pairs, so zip() is needed
     self._train_op = optimizer.apply_gradients(zip(grads, <list of variables>))
     ```

     此时就差一个<list of variables>不知道了，也就是需要对哪些variables进行求导,答案是：trainable variables:

     ```python
     tvars = tf.trainable_variables()
     ```

     此时再看官方PTBModel中的代码：

     ```python
     # 在运行过程中想要调整gradient值，就不能直接简单的optimizer.minimize(loss)而是要显式计算gradients
     self._lr = tf.Variable(0.0, trainable=False)
     tvars = tf.trainable_variables()
     grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                       config.max_grad_norm)
     optimizer = tf.train.GradientDescentOptimizer(self._lr)
     self._train_op = optimizer.apply_gradients(
         zip(grads, tvars),
         global_step=tf.contrib.framework.get_or_create_global_step())

     self._new_lr = tf.placeholder(
         tf.float32, shape=[], name="new_learning_rate")
     self._lr_update = tf.assign(self._lr, self._new_lr)
     ```

     其中**tf.clip_by_global_norm()**可用于用于控制梯度爆炸的问题。

     梯度爆炸和梯度弥散的原因一样，都是因为链式法则求导的关系，导致梯度的指数级衰减。为了避免梯度爆炸，需要对梯度进行修剪。详见[tensorflow笔记：多层LSTM代码分析 ](http://blog.csdn.net/u014595019/article/details/52759104)

2. main()

   main首先要读取并处理数据、配置模型并且控制模型运转。

   * 读取数据、设置config

     ```python
     # 在ptb_raw_data中已经将原始文本转换为id形式
     raw_data = reader.ptb_raw_data(FLAGS.data_path)
     train_data, valid_data, test_data, vocab_size = raw_data

     # 原始数据刚好是10000个单词，所以不需要修改config.vocab_size
     # 但是我有试过修改训练数据，所以加上了这句
     config = get_config()
     config.vocab_size=vocab_size

     eval_config = get_config()
     eval_config.batch_size = 1
     eval_config.num_steps = 1
     eval_config.vocab_size=vocab_size
     ```

     重点关注ptb_raw_data()方法。此方法中有几个关键步骤：

     * 根据训练数据构件单词表

       ```python
       word_to_id = _build_vocab(train_path)

       def _build_vocab(filename):
         """
         此方法读取原始数据，将换行符替换为<eos>，然后根据词频构件一个词汇表并返回。
         """
         data = _read_words(filename)

         counter = collections.Counter(data)
         count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

         words, _ = list(zip(*count_pairs))
         word_to_id = dict(zip(words, range(len(words))))

         return word_to_id

       def _read_words(filename):
         # 在这里讲换行符替换为了<eos>
         with tf.gfile.GFile(filename, "r") as f:
           return f.read().decode("utf-8").replace("\n", "<eos>").split()
       ```

       * 将原始train/valid/test数据转换为id形式

         根据上面得到的word_to_id词汇表对原始数据进行转化：

         ```python
         train_data = _file_to_word_ids(train_path, word_to_id)
         valid_data = _file_to_word_ids(valid_path, word_to_id)
         test_data = _file_to_word_ids(test_path, word_to_id)
         ```

   * 生成/训练模型

     以train模式为例：

     ```python
     with tf.name_scope("Train"):
         # PTBInput中根据config设置好batch_size等，还初始化了input(slice0)以及targetOutput(slice1)
         train_input = PTBInput(config=config, data=train_data, name="TrainInput")
         with tf.variable_scope("Model", reuse=None, initializer=initializer):
             m = PTBModel(is_training=True, config=config, input_=train_input)
         tf.scalar_summary("Training Loss", m.cost)
         tf.scalar_summary("Learning Rate", m.lr)
     ```

     基本是初始化模型的标准套路，但是需要注意PTBInput()

     在PTBInput中通过reader.ptb_producer()生成input和targets。

     ```python
     class PTBInput(object):
         """The input data."""

         def __init__(self, config, data, name=None):
             self.batch_size = batch_size = config.batch_size
             self.num_steps = num_steps = config.num_steps
             self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
             # input是当前slice[batchsize*num_steps]，output是下一个slice同样是[batchsize*num_steps]
             self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)
     ```

     在ptb_producer()中比较有趣的是最后几句：

     ```python
     def ptb_producer(raw_data, batch_size, num_steps, name=None):

       # 其他代码与注释

       i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
       x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
       y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
       return x, y
     ```

     i的本质是range_input_producer()获得的一个FIFOQueue.dequeue()(个人认为近似一个函数)，外部调用x和y时就可以通过i不断更新自身的值。因为本模型要做的是预测下一个词，所以在这里y(target)就是x(input)右移一位。

     tf中的**队列**和其他变量一样，是一种有状态的节点，其他节点可以把新元素插入到队列后端(rear)，也可以把队列前端(front)的元素删除。有如下例子：

     ```python
     q=tf.FIFOQueue(3,'float')
     init=q.enqueue_many(([0.,0.,0.],))

     x=q.dequeue()
     y=x+1
     q_inc=q.enqueue([y])

     # 注意，如果不写sess会报错
     with tf.Session() as sess:
         init.run()
         
         q_inc.run()
         q_inc.run()
         q_inc.run()
     ```

     在sess中从队列前端取走一个元素，加上1之后，放回队列的后端。慢慢地，队列的元素的值就会增加，示意图如下：

     ![img](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/IncremeterFifoQueue.gif)

     更多信息请参考[TensorFlow 官方文档中文版-线程和队列](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/threading_and_queues.html)，

     ​
     之后循环max_max_epoch次(文本重复次数)，循环过程中调整学习率，再调用run_epoch()训练模型。

     ```python
     with sv.managed_session() as session:
         for i in range(config.max_max_epoch):
             # 修改学习速率大小
             lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
             m.assign_lr(session, config.learning_rate * lr_decay)

             train_perplexity = run_epoch(session, m, eval_op=m.train_op,verbose=True)
     ```

   * run_epoch()

     首先设置需要run获取的数据，如果eval_op不为空，那么调用它并让模型根据预设代码自动优化。

     ```python
     fetches = {
         "cost": model.cost,
         "final_state": model.final_state,
     }
     if eval_op is not None:
         fetches["eval_op"] = eval_op

     for step in range(model.input.epoch_size):
         feed_dict = {}
         for i, (c, h) in enumerate(model.initial_state):
             feed_dict[c] = state[i].c
             feed_dict[h] = state[i].h

         vals = session.run(fetches, feed_dict)
         cost = vals["cost"]
         state = vals["final_state"]

         costs += cost
         iters += model.input.num_steps

         if verbose and step % (model.input.epoch_size // 10) == 10:
             print("%.3f perplexity: %.3f speed: %.0f wps" %
                   (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                    iters * model.input.batch_size / (time.time() - start_time)))
     return np.exp(costs / iters)
     ```























