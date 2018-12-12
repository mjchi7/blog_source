
<a href="https://colab.research.google.com/github/mjchi7/blog_source/blob/seq2seq_p1/2018_12_12_dynamic_rnn.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# What's `tf.nn.dynamic_rnn`, anyway?

This document is based largely on the excellent [source](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/) where it documents a lot of important tips and tricks. The problem I find is that it lacks a lengthy discussion of `tf.nn.dynamic_rnn`, and how to interpret its outputs, which is what this document hope to achieve.

## Recurrent Neural Network (RNN)
Recurrent Neural Network (RNN) is a class of neural network that excels at data that exhibit certain temporal pattern (such as sentences, audio signal, time series data, and etc.). The main idea behind RNN is that, say we have a sequence of data as such:
```python
data = ["Some", "random", "data", "."]
```
The data can be viewed in terms of `time_steps`, where the element `"Some"` can be refer to *input at time step 1*, $x_{t=1}$, `"random"` as $x_{t=2}$, `"data"` as $x_{t=3}$, and `"."` as $x_{t=4}$.

![figure-1](source\2018_12_12_dynamic_rnn\diag1_rnn_normal.png)  
<center>Figure 1: Basic RNN Operation.</center>


The RNN will take as input the first datum `"Some"`, pass it into the network, producing an output $y_{t=1}$ and hidden state representation $h_{t=2}$ using a set of weight $W_{rnn}$ and bias $b_{rnn}$. For the second data point `"random"`, the network will again use back the **same set of weight and bias, $W_{rnn}$ and $b_{rnn}$** to produce an output $y_{t=2}$ and hidden state representation $h_{t=2}$. This will go on until the hidden state and output for ${t=3}$ and $t=4$ are computed. If reader is interested to understand more on RNN, here is a very good [resource](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/).


## Let's implement it
The way to implement it, intuitively, is to conduct a for loop for how many time steps there is, in each time step we pass in the previous hidden states $h_{t-1}$ and current input $x_t$

```python
h_t = np.zeros(0)
for step in timestep:
	h_t_prev = h_t
	h_t = sigmoid(W_x * x_t + W_h * h_t_prev)
```
*Note: The equation aren't exactly complete in the sense that a bias term is purposely left out for the sake of clarity. The important thing to note here is how we are re-using the same weights when we go through the sequence*

This implementation is known as [**static unrolled rnn**](https://www.tensorflow.org/api_docs/python/tf/nn/static_rnn). It forces user to define how many time steps are there through the variable `timestep` prior to building the computation graph, which cause the following disadvantage:

####  Computation resources wastage 
Imagine if your training data contains 2 sentences of length 10 and 6. For example:

> **Sentence 1**: "I love pineapple on pizza regardless of what people say".  
> **Sentence 2**: "But pineapple on pizza is disgusting".

In order  to make the computation works, you will need to zero pad the second sentence of length 6 so that it has length of 10, as such:

> **Sentence 2**: "But pineapple on pizza is disgusting **<pad\> <pad\> <pad\> <pad\>**"

During forward propagation, since the computation graph have been defined strictly to loop for 10 times (to accommodate the longest sentence - in this case sentence 1), the **<pad\>** value in the second sentence will also undergo the same computation, which we know isn't necessary. In other word, the ideal case is to stop computing after the 6th word. 

### `tf.nn.dynamic_rnn` to the rescue!
*tf.nn.dynamic_rnn* is a way to counter the concern raised above: it will not compute the <pad\> value of sentence and just append vector of "0" so the matrix size agree.
In order to do that, tf.nn.dynamic_rnn requires an additional input: `sequence_length`

#### `sequence_length`
This parameter is an input of type `tf.int32` and of shape `[batch_size]`. The value in this parameter should tell the function *"how long is each sentence in this batch?"*. Continuing with our example above, for the two sentences, their respective `sequence_length` parameters should be 
```python
sequence_length = [10, 6]
```
With this parameter, the function `tf.nn.dynamic_rnn` will know for current sentence, when should it stop computing the hidden states and return the computed states, instead of continuing with the calculation for \<pad> value.

Keen reader by now should be wondering: What about the outputs of hidden states for the <pad\> value? The reason this question might pop up is because you understand that in some advance implementation, certain mechanism (such as Attention) requires all the hidden states value from all the time steps and the shape should agree regardless of the batch they belong to. In another words, taking our first sentence as example, in the end of the RNN forward propagation, we should have a hidden states Tensor coming out from all the time steps

**`[sentence_length, hidden_state_size]`**  
--- Diagram here to show the output of rnn cell from all the timesteps

What `tf.nn.dynamic_rnn` will do is to simply set the hidden state of skipped tokens (the <pad\> value it didn't calculate) to vectors of zeros of the size **`[hidden_state_size]`**

## In action 
To better visualize the matrices and their respective shape, let's code up a simple example:




```python
import tensorflow as tf
import numpy as np

# The dimension is in the form of
# (batch_size, time_steps, n_features)
# Where each batch element can be thought of as a sentence. 
# NOTE: Do not use random.randint32 as we are passing it through a non-linear function
#       Values higher than 1 will saturate the non-linear function easily. 
#       Recall how in normal application, the x input passed in will be normalized
#       word embeddings.
x = np.random.random(size=(2,10,5))
print(x)
```

    [[[0.62302861 0.01914905 0.17387512 0.22401975 0.80820405]
      [0.33198728 0.50174796 0.35421527 0.94233084 0.78846405]
      [0.82611666 0.30795712 0.98204377 0.71652138 0.52117634]
      [0.31457755 0.86598024 0.5040601  0.35834248 0.28594774]
      [0.41854428 0.12170197 0.99579171 0.65329134 0.61220468]
      [0.03912358 0.06279758 0.00299586 0.4851044  0.03213688]
      [0.68648015 0.16620824 0.92352208 0.42803549 0.06547576]
      [0.53588118 0.47669298 0.04755545 0.01279447 0.88792542]
      [0.40170098 0.48165938 0.75279648 0.31071382 0.10778935]
      [0.33770841 0.97168153 0.91277792 0.09921978 0.31437892]]
    
     [[0.63208416 0.69188769 0.38419977 0.29929609 0.35853236]
      [0.50405219 0.58771954 0.88271526 0.91690596 0.49007223]
      [0.7579435  0.7635813  0.25311605 0.8879893  0.5781744 ]
      [0.74723753 0.03005038 0.4157741  0.9749357  0.76512149]
      [0.83385818 0.27898849 0.86003361 0.7989492  0.09784158]
      [0.01292652 0.6188105  0.68942374 0.59556264 0.57880948]
      [0.37413241 0.35856435 0.72397363 0.64681828 0.78354511]
      [0.49312741 0.27576672 0.21304194 0.50086376 0.46582526]
      [0.18289339 0.79471481 0.05845358 0.29051561 0.5380547 ]
      [0.79893396 0.89251122 0.5873558  0.26831584 0.23983071]]]
    

From the console output, we can see that a toy example has been generated for the purpose of this demonstration. The tensor shape is of the following:  
(batch_size, time_steps, n_features)

Where **batch_size** can be thought of as the number of sentences we want to process in each sentence.
**time_steps** can be thought of as the number of words in each sentence and
**n_features** can be thought of as the embedding dimension of each words.


```python
# Let's assume here "-1" represents <pad> tokens
x[1, 6:] = -1
print(x)
```

    [[[ 0.62302861  0.01914905  0.17387512  0.22401975  0.80820405]
      [ 0.33198728  0.50174796  0.35421527  0.94233084  0.78846405]
      [ 0.82611666  0.30795712  0.98204377  0.71652138  0.52117634]
      [ 0.31457755  0.86598024  0.5040601   0.35834248  0.28594774]
      [ 0.41854428  0.12170197  0.99579171  0.65329134  0.61220468]
      [ 0.03912358  0.06279758  0.00299586  0.4851044   0.03213688]
      [ 0.68648015  0.16620824  0.92352208  0.42803549  0.06547576]
      [ 0.53588118  0.47669298  0.04755545  0.01279447  0.88792542]
      [ 0.40170098  0.48165938  0.75279648  0.31071382  0.10778935]
      [ 0.33770841  0.97168153  0.91277792  0.09921978  0.31437892]]
    
     [[ 0.63208416  0.69188769  0.38419977  0.29929609  0.35853236]
      [ 0.50405219  0.58771954  0.88271526  0.91690596  0.49007223]
      [ 0.7579435   0.7635813   0.25311605  0.8879893   0.5781744 ]
      [ 0.74723753  0.03005038  0.4157741   0.9749357   0.76512149]
      [ 0.83385818  0.27898849  0.86003361  0.7989492   0.09784158]
      [ 0.01292652  0.6188105   0.68942374  0.59556264  0.57880948]
      [-1.         -1.         -1.         -1.         -1.        ]
      [-1.         -1.         -1.         -1.         -1.        ]
      [-1.         -1.         -1.         -1.         -1.        ]
      [-1.         -1.         -1.         -1.         -1.        ]]]
    

Comparing the console output to previous console output, it is evident that plenty of "-1" tokens have been added to the second sentence to represents <pad\> value. Meaning that we are trying to illustrate the scenario where sentence two are shorter than sentence one, and are being padded with the token "-1".


```python
# Define the sequence length for each of our sentences above.
seq_length = [10, 6]
# Let's build our graph
#                      [batch_size, time_steps, n_features]
x_ph = tf.placeholder(tf.float32, [2, 10, 5])
seq_length_ph = tf.placeholder(tf.int32, [2,])

# We then proceed to define our RNN cell by supplying hidden_state_size of 3
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=3) 

# Building the dynamic rnn graph
outputs, states = tf.nn.dynamic_rnn(cell=basic_cell,
                                   inputs=x_ph,
                                   sequence_length=seq_length_ph,
                                   dtype=tf.float32)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs_val, states_val = sess.run([outputs, states], feed_dict={x_ph: x, seq_length_ph: seq_length})
  print(outputs_val)
  print()
  print()
  print(states_val)
```

    WARNING:tensorflow:From <ipython-input-3-609a96374344>:8: BasicRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.SimpleRNNCell, and will be replaced by that in Tensorflow 2.0.
    [[[-0.6043415  -0.766119    0.42360705]
      [-0.8556056  -0.5948331   0.5420409 ]
      [-0.7402248  -0.47115093  0.25555268]
      [-0.72596866 -0.18736169  0.46713755]
      [-0.5615917  -0.39451015 -0.05021511]
      [-0.3225749  -0.23166811 -0.06871123]
      [-0.67363656 -0.15471931 -0.08890712]
      [-0.61396563 -0.8143476   0.6512444 ]
      [-0.53082305  0.288834    0.45555303]
      [-0.5637323  -0.09094252  0.1782998 ]]
    
     [[-0.8005068  -0.5082021   0.39209768]
      [-0.83872294 -0.41006085  0.18253234]
      [-0.87609506 -0.81013316  0.3801973 ]
      [-0.7708805  -0.7730645   0.35276362]
      [-0.7557189  -0.25022933  0.09978031]
      [-0.74189085 -0.30529392  0.20443617]
      [ 0.          0.          0.        ]
      [ 0.          0.          0.        ]
      [ 0.          0.          0.        ]
      [ 0.          0.          0.        ]]]
    
    
    [[-0.5637323  -0.09094252  0.1782998 ]
     [-0.74189085 -0.30529392  0.20443617]]
    


### Understanding the outputs of `tf.nn.dynamic_rnn`

From the documentation, it stated that `tf.nn.dynamic_rnn` returns the following output:

> A pair (outputs, state) where:
> 
> **`outputs`**: The RNN output Tensor of shape **`[batch_size, max_time, cell.output_size]`** or **`[max_time, batch_size, cell.output_size]`** depending on your `time_major` parameter
>  
>  **`state`**: The final state, of shape **`[batch_size, cell.state_size]`**.


From the console output, we can observe the value of both `outputs` and `states` of `tf.nn.dynamic_rnn`. First thing to note here is that in  `outputs,` each row of vector represents the hidden state value produced by the RNN cell at each time steps. As specified in our code, the hidden state is of dimension of 3, which is why each word in our sentence actually produce a vector of 3 elements. In another word, for sentence 1, the computed hidden states at each time steps $t$ is as following (note: values extracted directly from the output):

$h_{t=1} = [-0.6043415,  -0.766119,    0.42360705]$  
$h_{t=2} =  [-0.8556056,  -0.5948331,   0.5420409 ]$  
$h_{t=3} = [-0.7402248,  -0.47115093,  0.25555268]$  
$h_{t=4} = [-0.72596866, -0.18736169,  0.46713755]$  
$h_{t=5} = [-0.5615917,  -0.39451015, -0.05021511]$  
$h_{t=6} = [-0.3225749,  -0.23166811, -0.06871123]$  
$h_{t=7} = [-0.67363656, -0.15471931, -0.08890712]$  
$h_{t=8} = [-0.61396563, -0.8143476,   0.6512444 ]$  
$h_{t=9} = [-0.53082305,  0.288834,    0.45555303]$  
$h_{t=10} = [-0.5637323,  -0.09094252,  0.1782998 ]$


Moving on to our second sentence, where we explicitly mention the sequence_length is 6, we can see that the words in sentence 2 are vector of zeros beyond the 6th item. That's because the function `tf.nn.dynamic_rnn` will only run the recurrent cell on the first 6 words of sentence 2, and for the rest of the sentence (7th element and beyond), it knows that they are just <pad\> value and no computation needs to be done, (with the information we give through `seq_length` variable.) hence the the rest of the hidden state representations are being zero-padded. Concretely, the hidden states for second sentence at each time steps are:  

$h_{t=1} = [-0.8005068  -0.5082021   0.39209768]$  
$h_{t=2} = [-0.83872294 -0.41006085  0.18253234]$  
$h_{t=3} = [-0.87609506 -0.81013316  0.3801973 ]$  
$h_{t=4} = [-0.7708805  -0.7730645   0.35276362]$  
$h_{t=5} = [-0.7557189  -0.25022933  0.09978031]$  
$h_{t=6} = [-0.74189085 -0.30529392  0.20443617]$  

Since we explicitly tell the function the second sentence is only up to $t=6$, it respond by only computing the hidden state representation up to $t=6$ and zero pad the rest.

Finally, looking at the printed value of `state`, we can see that it consists of the **last** hidden state representation of both sentence. In vanilla seq2seq, this is all we care about the output of encoder, which will be used as the initial hidden state of decoder to generate output. 


