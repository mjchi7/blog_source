
<a href="https://colab.research.google.com/github/mjchi7/blog_source/blob/master/_draft_seq2seq_minimal_p1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Seq2seq: The verbose, lengthy, detailed implementation; Part 1: Training seq2seq
I first learned about seq2seq in the amazing Stanford open course [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/). This is, hands down, one of the must-take courses if you are keen to understand deep learning better, particularly their applications in Natural Language Processing. The skillset they provided in this course is can be definitely carried over to other domain of deep learning such as in the domain of vision. They have also provided assignments for you to get your hands dirty to really understand what you learn in the class and how are they connected to the implementation. 

&nbsp;&nbsp;&nbsp;&nbsp;After this course is that I am quite convinced that I can articulate what a *seq2seq* model is, what they do, and how are they constructed, but one particular problem I faced is that after I fired up my Python program and imported the Tensorflow package try to try my hands on implementing the *seq2seq* model, I get stucked after every few lines and cannot proceed beyond creating an embedding layer. Perhaps I have missed some of the important points throughout my journey in CS224N, but it just seems like there is this big gap between learning from classes and to really implement them on your own (*disclaimer: I have not complete assignment 4, maybe that's why*). Looking at the tutorials and guide outside, a lot of the times the function they used are some really high level function that hides away a lot of the details. This is a problem for me because to really appreciate the convenience provided by those high level functions, I will need to see for myself what do the low-level functions lacks. 

&nbsp;&nbsp;&nbsp;&nbsp;One prime example I came across is the convenient `Decoder` object provided by Tensorflow, together with a `TrainingHelper` object, just define your parameters correctly and it shall decode nicely. It has been an immense challenge to wrap my tiny head around this set of objects and functions as I am used to seeing the "decoding" process in a step by step manners, where you get the final hidden state from encoder, feed it into the decoder as initial hidden state along with some indicative tokens such as `<start>` which shall serve as the first input to decoder. Depending on whether the model is in training or inference mode, we will feed either the label token or the previously predicted token into the decoder as the input for $t=2$, $t=3$, and so on. Therefore, I have decided to try and implement on my own, with the help from several "minimal" *seq2seq* implementation from other authors to better understand the whole process. 


Summarizing the text above, the purpose of this document is twofold: 
1. It serves to demonstrate the implementation of **seq2seq** model with the least convoluted functions available in Tensorflow pacakges.
2. The explanations of code (and sometimes even the code themselves) are purposely made as verbose in order for people who is coming directly from a theory class to connect better. 

Let's get started!

## Importing necessary packages
As usual, the first step is to identify the packages we need and importing them


```python
import numpy as np
import tensorflow as tf
```

## Creating dataset to work with
No doubt the *seq2seq* model can process datasets that are complex in nature (such as news, articles, and etc.). However since clarity is the main focus in this demonstration, a very simple Question and Answering dataset has been artificially created and our model will learn these two examples only. For the sake of clarity, I have manipulate the string data manually so that it can be shown what we are dealing with. The purpose is for user not to dwell too much on data preparation, and just get over it quickly and head straight to *seq2seq* implementation. However, it is impossible to understand the implementation if you do not grasp what data we are processing firmly, hence the simple data example as well as the "stupid" data processing process.

Let's start with two very simple question and answering data set

> Q1: "What is the color of banana?"
> A1: "Yellow most of the time."

> Q2: "What makes up water molecules?"
> A2: "Hydrogen and oxygen."

The goal is to have the machine learn how to answer these two questions (**q**) with the desired answers (**a**). Puncutations and capitalization are ignored in this example.

A high level goal is to first create a dictionary so that it contains all the unique words in our dataset, and then we will tokenize the sentence into a list of individual words. After that, each word will be mapped into their respective index based on our dictionary so that we end up with an arrays of number. In addition to that, we will add several special tokens which serve different purposes:

> **<q\>**: The start token for a "question" sentence.  
> **<a\>**: The start token for an "answer" sentence.  
> **<pad\>**: The padding token for shorter sentence.   

Sometimes people (me included) might wonder why do we need the **<pad\>** token for? Remember to know the distinction between Python's list and numpy array: You can have list of list of different element, but you cannot have matrix with different columns in diffferent rows (and vice versa). Here is what I meant:



```python
q = [[1,2,3],[4,5]]
q_arr = np.array(q)
print('type:', type(q_arr))
print(q_arr)
print('type of q_arr[0]:', type(q_arr[0]))
```

    type: <class 'numpy.ndarray'>
    [list([1, 2, 3]) list([4, 5])]
    type of q_arr[0]: <class 'list'>
    

From the console output, we can see that instead of converting it into a matrix (array of array), it is being convereted into an **array of list**. Compare that with the following example:


```python
q = [[1,2,3],[4,5,6]]
q_arr = np.array(q)
print('type:', type(q_arr))
print(q_arr)
print('type of q_arr[0]:', type(q_arr[0]))
```

    type: <class 'numpy.ndarray'>
    [[1 2 3]
     [4 5 6]]
    type of q_arr[0]: <class 'numpy.ndarray'>
    

Where it shows we have **array of array**. Therefore, it is important to pad the sentence so that all of them have the same sentence length such that they can be converted into a legitimate tensor later on.

Additionally, we will need to have a set of parameters to specify the `sentence_length` of both `q` and `a`. The usage of `sentence_lenght` will become evident later but at the time being, you can think of them as information that we will pass to our *seq2seq* model to notify them when should they stop processing the unnecessary tokens in each sentence (read: **<pad\>**).



```python
# Define our data
# We want to train our machine to answer this simple question

# Q1: "What is the color of banana?" A1: "yellow most of the time"
# Q2: "What makes up water molecules?" A2: "Hydrogen and oxygen"

# In this simple example, we will be ignoring the punctuations

# Let's define our simple dictionary
word2idx = {'<pad>':0, 'what': 1, 'is': 2, 'the': 3, 'color': 4, 'of': 5, 'banana': 6,
           'yellow':7, 'most': 8, 'time': 9, 'makes': 10, 'up': 11, 'water': 12, 'molecules':13,
           'hydrogen': 14,'and':15, 'oxygen': 16, '<q>': 17, '<a>': 18}
# As usual, the reverse dictionary is needed so we can map our prediction back to word
# for illustration later.
idx2word = {v:k for k, v in word2idx.items()}

# With that, let's tokenize the Q1, Q2, A1, and A2
# Do note that the tokenization can be very easily done with simple python built-in
# string manipulation function (strip, split, etc.) But it is being manually typed
# out for the sake of simplicity.

q = [['<q>', 'what', 'is', 'the', 'color', 'of', 'banana'], ['<q>', 'what', 'makes', 'up', 'water','molecules']]
a = [['<a>', 'yellow', 'most', 'of', 'the', 'time'], ['<a>', 'hydrogen', 'and', 'oxygen']]

# It is needed, however, to have the questions and answers to have the same length
#     in order to cast them into numpy array later on (doesn't make sense for matrix)
#     to have different column based on different rows.
# We add <pad> token to fulfil this requirement
q_raw = [['<q>', 'what', 'is', 'the', 'color', 'of', 'banana'], ['<q>', 'what', 'makes', 'up', 'water','molecules', '<pad>']]
a_raw = [['<a>', 'yellow', 'most', 'of', 'the', 'time'], ['<a>', 'hydrogen', 'and', 'oxygen', '<pad>', '<pad>']]

# In addition to the q and a, we will need to identify the length of each sequence
# since we will be using tf.nn.dynamic_rnn later (more on this later).
q_seq_length = [7,6]
a_seq_length = [6,4]

# Let's map our q_raw and a_raw to their respective indices
# To do so, we make use of the numpy's vectorize function
mapper = np.vectorize(word2idx.get)
# The code above create a "vectorized" function, that will map each element in 
#    a given nested structure and apply the function word2idx.get to it
q_tensor = mapper(q_raw)
a_tensor = mapper(a_raw)

# Let's print the final dataset to see what we will be dealing with using our 
# seq2seq machine
print('q_tensor:\n', q_tensor)
print('a_tensor:\n', a_tensor)

```

    q_tensor:
     [[17  1  2  3  4  5  6]
     [17  1 10 11 12 13  0]]
    a_tensor:
     [[18  7  8  5  3  9]
     [18 14 15 16  0  0]]
    

As you can see from the console output, both `q_tensor` and `a_tensor` has been successfully converted to their respective index according to the dictionary `word2idx`

## Seq2seq engine
For the subsequent part, we will mainly, be implementing the whole seq2seq mechanism using tensorflow. The implementation will be as verbose as possible (read: lengthy, but clear) to allow people familiar with the theory of seq2seq to better see how it is being connected to the process we studied in class. 

The whole graph building process will be divided into 4 parts: 

1. **Preliminaries**: declares several important parameters
2. **Encoder**: we build the forward propagation for encoder
3. **Decoder**: we build the forward propagation for decoder
4. **Misc**: the loss calculation, optimizers, and etc.



### Part 1: Preliminaries
In this part, we will define several important parameters for our seq2seq machine.

> 1. Constants  
These constants are basically some numbers that define our *seq2seq* models. Constants such as the `vocab_size`, `embed_dim` are all neede during computational graph construction time so that the weight matrices can have the correct size and shape.

> 2. Initializers  
Initializers are needed when we construct variables in Tensorflow as a mean for us to initialize the variables that will subsequently be updated during training phase.

> 3. Placeholders  
Placecholders can be thought of as an "entry point" for user to feed in data into the model. The obvious placeholders including `q_ph` and `a_ph` is needed so that we can feed in our `q_tensor` and `a_tensor` data into the graph when we run it.

> 4. Embedding layers  
As usual for all NLP task, embedding layer is needed to map the word index into their respective [embedding vectors](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa). Usually a pre-trained embedding layers will be used (such as Skip-gram, Bag of words, GloVe, and etc.), but in this example we will just train it together with our task on hand.


```python
# Constants
vocab_size = len(word2idx.keys()) # The number of unique word in our whole dictionary
embed_dim = 5 # The dimension of our embedded words
hidden_state_size = 32 # The size of our RNN cell 
q_max_timesteps = max(q_seq_length) # The length of longest sentence in 'q'
a_max_timesteps = max(a_seq_length) # The length of longest sentence in 'a'

# Useful initializers
# xavier initializer promises a more stable gradient (for more read: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
xav_init = tf.contrib.layers.xavier_initializer()
# zero initializer initialize a variable with all zeros
zero_init = tf.zeros_initializer()

# Placeholders
# Note: We specify "None" for batch_size so that we can re-use the same 
#       placeholder during inference time without having to force our 
#       'q' to have the same batch_size as they are in training time.
q_ph = tf.placeholder(tf.int32, [None, q_max_timesteps])
q_seq_length_ph = tf.placeholder(tf.int32, [None,])
a_ph = tf.placeholder(tf.int32, [None, a_max_timesteps])
a_seq_length_ph = tf.placeholder(tf.int32, [None,])

# Embedding layer (since both encoders and decoders are sharing the same embedding
# layer, we put it in "preliminaries step")
embedding_layers = tf.get_variable('embedding_layer',
                                  shape=(vocab_size, embed_dim),
                                  dtype=tf.float32,
                                  initializer=xav_init)
```

### Part 2: Encoder
In this part, we will build the forward propagation for the encoder model. In the gist of it, encoder takes in a sequence of words (in their embedding vector form), producing a `hidden_state` at each timesteps and in the end, producing a final `hidden_state` which intuitively, contains all the information in the entire sequence. In the end of this encoder, we are expecting the final `hidden_state` of encoder which will be used as a "seed" at decoder to generate output sequence.

#### `tf.nn.dynamic_rnn` 
Before we move further, it is imperative to talk a bit about the function `tf.nn.dynamic_rnn`. Simply put, `tf.nn.dynamic_rnn` will *dynamically* run our flow graph based on the parameter `sequence_length`. If we recall from our toy data example, our *q2* is much shorter than *q1*, which we have padded with the token **<pad\>** in order for the tensor size to agree. We know that it isn't necessary to run our RNN cell over the **<pad\>**. For more detailed breakdown of this function, refers to **THIS BLOGPOST**

To construct the encoder, we will need an RNN cell, which comes in different flavours such as [`GRUCell`](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/GRUCell), [`LSTMCell`](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/LSTMCell), and [etc](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell). In this simple example we will just use a vanilla [`BasicRNNCell`](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/BasicRNNCell). `BasicRNNCell` implements the simplest form of Recurrent Neural Network: A set of weight W and bias b will be shared as it proceed in sequence, producing `hidden_state` at every timesteps.



```python
# 1. Before anything, we wil need a RNN cell to work with.
enc_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_state_size)

# 2. Mapping input word index to their respective embedded vectors
#     `enc_embed_inp` shape = [batch_size, max_timesteps, embed_dim]
enc_embed_inp = tf.nn.embedding_lookup(embedding_layers, q_ph)

# 3. The forward propagation for encoder
#    tf.nn.dynamic_rnn produces `hidden_state` for `inputs` based on `sequence_length`.
enc_out, enc_state = tf.nn.dynamic_rnn(enc_cell,
                                   inputs=enc_embed_inp,
                                   sequence_length=q_seq_length_ph,
                                   dtype=tf.float32,
                                   scope='encoder_rnn')
```

    WARNING:tensorflow:From <ipython-input-6-263895fcb771>:1: BasicRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.SimpleRNNCell, and will be replaced by that in Tensorflow 2.0.
    

### Part 3: Decoder
In this part, we will construct the foward propagation for the decoder. The initial state of the decoder will be the final state output from the encoder.  

For decoder, the process are different depends on whether we are training the model or making inference. In this part we wants to focus on the training process of decoder, meaning that at each timesteps, instead of feeding previously predicted output $\hat{y}_{t-1}$ as input at the next time step $x_{t}$, we take the ground truth label from `a_ph` and feed it as input.

At each timesteps, using the `hidden_states` output by the RNN cell, we will "project" it into the vocab space by putting a feed forward neural network there, which enable us to project a tensor of shape `hidden_state_size` to `vocab_size`. After the projection, softmax can be applied and using function `argmax`, we can get the index of the decoded word.



```python
# Sequence A: Training process
# 1. As usual, we will need to define our RNN cell.
dec_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_state_size)

# 2.  Training's decoder
dec_embed_inp = tf.nn.embedding_lookup(embedding_layers, a_ph)

# 3. Decoder's forward propagation during training phase
dec_out, dec_state = tf.nn.dynamic_rnn(cell=dec_cell,
                                      inputs=dec_embed_inp,
                                      sequence_length=a_seq_length_ph,
                                      initial_state=enc_state,
                                      scope='decoder_rnn')
# Note that in this case, we explicitly feed in our "dec_embed_inp" to the dynamic
# RNN as a way to "force teach" the network.
# 4. We need to project the hidden states output from decoder at each time steps 
#    to vocab_size 
W_proj = tf.get_variable('dec_out_proj',
                        shape=(hidden_state_size, vocab_size),
                        initializer=xav_init)
b_proj = tf.get_variable('dec_out_proj_bias',
                        shape=(vocab_size,),
                        initializer=zero_init)

# 5. Before we can multiply, we need to reshape our tensor
#    tf.matmul only allows matrix multiplication. So
#    we collapse our batch_size dimension.
dec_out_shaped = tf.reshape(dec_out, [-1, hidden_state_size])
proj_mul = tf.matmul(dec_out_shaped, W_proj) + b_proj
proj_shaped = tf.reshape(proj_mul, [-1, a_max_timesteps, vocab_size])

# At this point, the calculations are enough for us to start training the model.
# However, since we want to have a firmer grasp on the whole process, we shall
# output the "predicted" answer at each iterations. Therefore the following
# step is needed in order to turn our projections from decoder's hidden state 
# into our dictionary index.

# 6. Finally, to visualize how the output are being optimized while they are training,
#    we convert the projection to our vocabulary index.
proj_sm = tf.nn.softmax(proj_shaped)
pred = tf.argmax(proj_sm, axis=-1)
```

### Part 4: Loss and training optimization
In this part, the loss calculation will be defined with some masking techniques, so that loss made beyond sequence length will not matter in our objective.

The thing to note here is the masking tricks used to ignore errors made on the <pad\> error. Other than that it is a plain simple softmax cross entropy loss scalculation.

Finally we add an optimizer which will reduce the `batch_loss` against all the variables.


```python
# tf.nn.sparse_softmax_cross_entropy_with_logits allows label to be index instead of one hot version
# Note: the function tf.nn.sparse_softmax_cross_entropy_with_logits requires the input `logits` to be
#       un-softmaxed version. (the softmax operation will be carried out in the function itself.) 
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=a_ph, logits=proj_shaped)

# Masking techniques are employed here to prevent error made on <pad> to contribute
# to the total loss
mask = tf.sequence_mask(a_seq_length_ph, a_max_timesteps)
masked_loss = tf.boolean_mask(loss, mask)
batch_loss = tf.reduce_mean(masked_loss)

# Optimizers
# We are not doing too much modifications with our optimizer, just the default 
# setting will do.
opt = tf.train.AdamOptimizer().minimize(batch_loss)
```

    /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/gradients_impl.py:112: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
    

## Running the session

In this session, we will run our seq2seq training sequences. The training process have been made verbose so that user can be clear on how the training actually goes and to better visualize what were being produced at each different steps of our flow graph.

We first plug in our data through placeholder, creating a `feed_dict` which will be passed to the function `sess.run()`. The first argument of the function `sess.run()` contains a lot of the Tensor operation so that we can print the output of some intermediate operation that can better help user to visualize what we get at the end of each operation. In normal implementation, it is not necessary, as having the value of `batch_loss` is sufficient for user to monitor the training progress already.


```python
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  feed_dict = {q_ph: q_tensor,
               q_seq_length_ph: q_seq_length,
               a_ph: a_tensor,
               a_seq_length_ph: a_seq_length}
  
  for i in range(100):
    loss_v, mask_v, masked_loss_v, batch_loss_v, _, pred_v = sess.run([loss, mask, masked_loss, batch_loss, opt, pred], feed_dict=feed_dict)
    if i % 10 == 0:
      print("="*80)
      print('iter: {}'.format(i))
      print("="*80)
      print('loss_v:')
      print('shape', loss_v.shape)
      print('value:')
      print(loss_v)
      print('\n')
      print('mask_v')
      print('shape', mask_v.shape)
      print('value:')
      print(mask_v)
      print('\n')
      print('masked_loss_v')
      print('shape', masked_loss_v.shape)
      print('value:')
      print(masked_loss_v)
      print('\n')
      print('batch_loss_v')
      print('shape', batch_loss_v.shape)
      print('value:')
      print(batch_loss_v)
      print('\n')
      print('pred_v')
      print('shape', pred_v.shape)
      print('value:')
      print(pred_v)
      print('\n')
      print("Translated pred_v at current iter")
      for sent_tup, m in zip(enumerate(pred_v), mask_v):
        # Masking operation to remove the last 2 elements of a2
        # `sent_tup` is a tuple with two items: (`sent_no`, `sent`)
        # where `sent_no` is from the function enumerate and `sent` is from 
        # iterating `pred_v`
        sent_no = sent_tup[0]
        sent = sent_tup[1]
        sent_masked = [s for s,m in zip(sent, m) if m] 
        sentence = ''
        for w in sent_masked:
          sentence = sentence + idx2word.get(w) + ' '
        print("sentence {}: {}".format(sent_no, sentence))

```

    ================================================================================
    iter: 0
    ================================================================================
    loss_v:
    shape (2, 6)
    value:
    [[2.9701633 2.914389  3.4912496 3.7816746 2.4590976 2.8373427]
     [2.8659377 2.915552  3.2338321 2.6917157 2.944439  2.944439 ]]
    
    
    mask_v
    shape (2, 6)
    value:
    [[ True  True  True  True  True  True]
     [ True  True  True  True False False]]
    
    
    masked_loss
    shape (10,)
    value:
    [2.9701633 2.914389  3.4912496 3.7816746 2.4590976 2.8373427 2.8659377
     2.915552  3.2338321 2.6917157]
    
    
    batch_loss_v
    shape ()
    value:
    3.0160954
    
    
    pred_v
    shape (2, 6)
    value:
    [[ 9  5 17 16  3  8]
     [11  5  1 13  0  0]]
    
    
    Translated pred_v at current iter
    sentence 0: time of <q> oxygen the most 
    sentence 1: up of what molecules 
    ================================================================================
    iter: 10
    ================================================================================
    loss_v:
    shape (2, 6)
    value:
    [[2.4268906 2.4414694 2.8275084 3.2480876 2.0690553 2.3562431]
     [2.4897208 2.3581908 2.7629275 2.1755989 2.95393   2.95393  ]]
    
    
    mask_v
    shape (2, 6)
    value:
    [[ True  True  True  True  True  True]
     [ True  True  True  True False False]]
    
    
    masked_loss
    shape (10,)
    value:
    [2.4268906 2.4414694 2.8275084 3.2480876 2.0690553 2.3562431 2.4897208
     2.3581908 2.7629275 2.1755989]
    
    
    batch_loss_v
    shape ()
    value:
    2.5155692
    
    
    pred_v
    shape (2, 6)
    value:
    [[18  7  1 16  3  9]
     [ 3 14  1 16  8  8]]
    
    
    Translated pred_v at current iter
    sentence 0: <a> yellow what oxygen the time 
    sentence 1: the hydrogen what oxygen 
    ================================================================================
    iter: 20
    ================================================================================
    loss_v:
    shape (2, 6)
    value:
    [[1.8450881 2.0092592 2.168913  2.7727945 1.7452543 1.9935005]
     [2.0221167 1.8691208 2.3140457 1.6472244 2.9632864 2.9632864]]
    
    
    mask_v
    shape (2, 6)
    value:
    [[ True  True  True  True  True  True]
     [ True  True  True  True False False]]
    
    
    masked_loss
    shape (10,)
    value:
    [1.8450881 2.0092592 2.168913  2.7727945 1.7452543 1.9935005 2.0221167
     1.8691208 2.3140457 1.6472244]
    
    
    batch_loss_v
    shape ()
    value:
    2.0387318
    
    
    pred_v
    shape (2, 6)
    value:
    [[18  7  8 16  3  9]
     [18 14  8 16  8  8]]
    
    
    Translated pred_v at current iter
    sentence 0: <a> yellow most oxygen the time 
    sentence 1: <a> hydrogen most oxygen 
    ================================================================================
    iter: 30
    ================================================================================
    loss_v:
    shape (2, 6)
    value:
    [[1.238597  1.531639  1.6141139 2.4497957 1.3561397 1.6419129]
     [1.4951814 1.546345  1.912676  1.2521341 2.9722102 2.9722102]]
    
    
    mask_v
    shape (2, 6)
    value:
    [[ True  True  True  True  True  True]
     [ True  True  True  True False False]]
    
    
    masked_loss
    shape (10,)
    value:
    [1.238597  1.531639  1.6141139 2.4497957 1.3561397 1.6419129 1.4951814
     1.546345  1.912676  1.2521341]
    
    
    batch_loss_v
    shape ()
    value:
    1.6038536
    
    
    pred_v
    shape (2, 6)
    value:
    [[18  7  8 16  3  9]
     [18 14 15 16 15 15]]
    
    
    Translated pred_v at current iter
    sentence 0: <a> yellow most oxygen the time 
    sentence 1: <a> hydrogen and oxygen 
    ================================================================================
    iter: 40
    ================================================================================
    loss_v:
    shape (2, 6)
    value:
    [[0.7772187 1.1304305 1.2213583 2.135445  1.0077025 1.2293954]
     [1.057499  1.2918723 1.5504545 1.0032358 2.9804475 2.9804475]]
    
    
    mask_v
    shape (2, 6)
    value:
    [[ True  True  True  True  True  True]
     [ True  True  True  True False False]]
    
    
    masked_loss
    shape (10,)
    value:
    [0.7772187 1.1304305 1.2213583 2.135445  1.0077025 1.2293954 1.057499
     1.2918723 1.5504545 1.0032358]
    
    
    batch_loss_v
    shape ()
    value:
    1.2404611
    
    
    pred_v
    shape (2, 6)
    value:
    [[18  7  8 16  3  9]
     [18 14 15 16  5  5]]
    
    
    Translated pred_v at current iter
    sentence 0: <a> yellow most oxygen the time 
    sentence 1: <a> hydrogen and oxygen 
    ================================================================================
    iter: 50
    ================================================================================
    loss_v:
    shape (2, 6)
    value:
    [[0.49477655 0.8751553  0.9037799  1.701108   0.775553   0.8721707 ]
     [0.8054055  0.98945266 1.2116487  0.8231473  2.9878733  2.9878733 ]]
    
    
    mask_v
    shape (2, 6)
    value:
    [[ True  True  True  True  True  True]
     [ True  True  True  True False False]]
    
    
    masked_loss
    shape (10,)
    value:
    [0.49477655 0.8751553  0.9037799  1.701108   0.775553   0.8721707
     0.8054055  0.98945266 1.2116487  0.8231473 ]
    
    
    batch_loss_v
    shape ()
    value:
    0.9452197
    
    
    pred_v
    shape (2, 6)
    value:
    [[18  7  8 16  3  9]
     [18 14 15 16  5  5]]
    
    
    Translated pred_v at current iter
    sentence 0: <a> yellow most oxygen the time 
    sentence 1: <a> hydrogen and oxygen 
    ================================================================================
    iter: 60
    ================================================================================
    loss_v:
    shape (2, 6)
    value:
    [[0.32474393 0.674966   0.670608   1.2956378  0.6085225  0.63341564]
     [0.65432954 0.71997255 0.94505185 0.6320225  2.9944944  2.9944944 ]]
    
    
    mask_v
    shape (2, 6)
    value:
    [[ True  True  True  True  True  True]
     [ True  True  True  True False False]]
    
    
    masked_loss
    shape (10,)
    value:
    [0.32474393 0.674966   0.670608   1.2956378  0.6085225  0.63341564
     0.65432954 0.71997255 0.94505185 0.6320225 ]
    
    
    batch_loss_v
    shape ()
    value:
    0.715927
    
    
    pred_v
    shape (2, 6)
    value:
    [[18  7  8  5  3  9]
     [18 14 15 16  5  5]]
    
    
    Translated pred_v at current iter
    sentence 0: <a> yellow most of the time 
    sentence 1: <a> hydrogen and oxygen 
    ================================================================================
    iter: 70
    ================================================================================
    loss_v:
    shape (2, 6)
    value:
    [[0.22401169 0.5017901  0.5103989  0.95941585 0.46851054 0.47186723]
     [0.52344847 0.533746   0.7481553  0.46270847 3.0003364  3.0003364 ]]
    
    
    mask_v
    shape (2, 6)
    value:
    [[ True  True  True  True  True  True]
     [ True  True  True  True False False]]
    
    
    masked_loss
    shape (10,)
    value:
    [0.22401169 0.5017901  0.5103989  0.95941585 0.46851054 0.47186723
     0.52344847 0.533746   0.7481553  0.46270847]
    
    
    batch_loss_v
    shape ()
    value:
    0.5404053
    
    
    pred_v
    shape (2, 6)
    value:
    [[18  7  8  5  3  9]
     [18 14 15 16  5  5]]
    
    
    Translated pred_v at current iter
    sentence 0: <a> yellow most of the time 
    sentence 1: <a> hydrogen and oxygen 
    ================================================================================
    iter: 80
    ================================================================================
    loss_v:
    shape (2, 6)
    value:
    [[0.16510466 0.3734976  0.40169114 0.6857595  0.35803232 0.35832983]
     [0.41854984 0.39872035 0.5911982  0.34295377 3.0054152  3.0054152 ]]
    
    
    mask_v
    shape (2, 6)
    value:
    [[ True  True  True  True  True  True]
     [ True  True  True  True False False]]
    
    
    masked_loss
    shape (10,)
    value:
    [0.16510466 0.3734976  0.40169114 0.6857595  0.35803232 0.35832983
     0.41854984 0.39872035 0.5911982  0.34295377]
    
    
    batch_loss_v
    shape ()
    value:
    0.4093837
    
    
    pred_v
    shape (2, 6)
    value:
    [[18  7  8  5  3  9]
     [18 14 15 16  5  5]]
    
    
    Translated pred_v at current iter
    sentence 0: <a> yellow most of the time 
    sentence 1: <a> hydrogen and oxygen 
    ================================================================================
    iter: 90
    ================================================================================
    loss_v:
    shape (2, 6)
    value:
    [[0.1285758  0.2848497  0.3229015  0.48550546 0.27528042 0.2782205 ]
     [0.34054768 0.30051303 0.4658513  0.25856727 3.0097823  3.0097823 ]]
    
    
    mask_v
    shape (2, 6)
    value:
    [[ True  True  True  True  True  True]
     [ True  True  True  True False False]]
    
    
    masked_loss
    shape (10,)
    value:
    [0.1285758  0.2848497  0.3229015  0.48550546 0.27528042 0.2782205
     0.34054768 0.30051303 0.4658513  0.25856727]
    
    
    batch_loss_v
    shape ()
    value:
    0.31408125
    
    
    pred_v
    shape (2, 6)
    value:
    [[18  7  8  5  3  9]
     [18 14 15 16  5  5]]
    
    
    Translated pred_v at current iter
    sentence 0: <a> yellow most of the time 
    sentence 1: <a> hydrogen and oxygen 
    

### Inspecting the training process

There are several things to notice here:

1. First, compare the matrix of `loss_v` and `masked_loss_v`. We can observe that after the masking operation, the last 2 loss value has been missing, which are actually the loss value of **<pad\>** token. Notice how it corresponds to the value of `mask_v`. 
2. Notice how in the "Translated pred_v at current iter", the last 2 word isn't being printed out. That's because we are not interested in the padding value prediction anyway, so I have purposely masked it using the same `mask_v` value.
3. Finally, notice how the "predicted sentence" gets closer to our ground truth label as the iteration number increases, and the `batch_loss_v` value decreases, which indicate that we are fitting our model to the data given. 

*Note: The model is hopelessly overfitting to the data for the sake of demonstration.*

## Summary

In this part, we have implemented the process to train our *seq2seq* model using the (hopefully) least convoluted function we have to really see things for ourselves. Keen reader might notice that our decoder requires input at each time steps in order to generate output. We know that in normal use case, all the decoder need is a "start token" and the final `hidden_state` from encoder to start generating output. In the next part, we will see how we can implement the *seq2seq* so that it can carry out inference mode with minimal changes to our current code.
