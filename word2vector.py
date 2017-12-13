
import numpy as np
import tensorflow as tf

"""create word vectors using skip gram model"""
"""argument 'one_hots' is the list of one hot vectors
'word_labels' is the one hot vecotrs respective one hot labels dependant on window size."""
def skip_gram(one_hots, word_labels, dimensions = 100):
    
    inputs = tf.placeholder(tf.float32, name = 'one_hots')
    outputs = tf.placeholder(tf.float32, name = 'labels')
    with tf.variable_scope('hidden_layer'):
        """this is the 1st set of weights with 100 features/dimensions per word vector"""
        weights = tf.get_variable('weights', shape = [one_hots.shape[1], dimensions], 
            initializer=tf.contrib.layers.xavier_initializer() ) 
        
        bias = tf.get_variable('bias3', dtype = tf.float32, 
        initializer = tf.random_normal([1]))  

        """multiplies our one hot vecotrs with our weights and adds bias"""
        """This layer puts our one hot vectors into their respective vector space"""
        hidden_layer = tf.matmul(inputs, weights) + bias 
        
        """2nd set of weights that puts our vector speaces into back into the amount of words
        in our vocabulary so we can get propper probabilites for our corpus"""
        weights2 = tf.get_variable('weights2', shape = [dimensions, one_hots.shape[1]], 
            initializer=tf.contrib.layers.xavier_initializer() ) 
        # did some wierd list comp below. fix that shit
        """This layer sets our nodes to the same amount as our intial one_hot vector inputs
        This is so we can put the output layer ibnto propper probability form"""
        output_layer = tf.matmul(hidden_layer, weights2) 
  
    with tf.variable_scope('errors'): 
        #probabilities = tf.nn.softmax(hidden_layer)
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits = output_layer, labels = outputs, name = 'error'))
        
        learn_word_vectors = tf.train.GradientDescentOptimizer(0.1).minimize(error)
        
        feed_dict = {inputs : one_hots, outputs: word_labels}
        with tf.Session() as sess:
            sess.run(weights.initializer)
            sess.run(bias.initializer)
            sess.run(weights2.initializer)
            iterations = 201
            for i in range(iterations):
                sess.run(learn_word_vectors, feed_dict)
                
                if i % 20 == 0:
                    print(sess.run(error,feed_dict))
                    
            if i == iterations - 1:
                """this returns our word vectors"""               
                return sess.run(hidden_layer, feed_dict)













