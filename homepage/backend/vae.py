from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from tensorflow.examples.tutorials.mnist import input_data
import os, pickle, sys

mnist = input_data.read_data_sets("MNIST", one_hot=False)
tf.reset_default_graph()
batch_size = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name="X")
Y = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name="Y")
Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name="keep_prob") #Used for dropout layers
number_of_latent_variables = 20
dec_in_channels = 1
reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels / 2

    #return mnist, batch_size, X_in, Y_flat, keep_prob, number_of_latent_variables, reshaped_dim, inputs_decoder

def leaky_relu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def encoder(X_in, keep_prob):
    
    with tf.variable_scope("encoder", reuse=None):
        X_reshaped = tf.reshape(X_in, shape=[-1, 28, 28, 1]) #reshape where -1 means the batch_size, 
                                                            #28 the height, 28 the width and 1 the number of channels!
        X = tf.layers.conv2d(X_reshaped, filters=64, kernel_size=4, strides=2, padding="same", activation=leaky_relu)
        X = tf.nn.dropout(X, keep_prob)
        X = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding="same", activation=leaky_relu)
        
        X = tf.nn.dropout(X, keep_prob)
        X = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=1, 
                             padding="same", activation=leaky_relu)
        X = tf.nn.dropout(X, keep_prob)
        X = tf.contrib.layers.flatten(X)
        mean = tf.layers.dense(X, units=number_of_latent_variables)
        sd = 0.5 * tf.layers.dense(X, units=number_of_latent_variables)
        epsilon = tf.random_normal([tf.shape(X)[0], number_of_latent_variables])
        z = mean + tf.multiply(epsilon, tf.exp(sd)) #tf.multiply is elementwise or Hadamard multiplication!
        
        return z, mean, sd
    
def decoder(sampled_z, keep_prob):
    
    with tf.variable_scope("decoder", reuse=None):
        X = tf.layers.dense(sampled_z, units=inputs_decoder, activation=leaky_relu)
        X = tf.layers.dense(X, units=inputs_decoder * 2 + 1, activation=leaky_relu)
        X = tf.reshape(X, reshaped_dim)
        X = tf.layers.conv2d_transpose(X, filters=64, kernel_size=4, strides=2, 
                             padding="same", activation=tf.nn.relu)
        X = tf.nn.dropout(X, keep_prob)
        X = tf.layers.conv2d_transpose(X, filters=64, kernel_size=4, strides=1, 
                             padding="same", activation=tf.nn.relu)
        X = tf.nn.dropout(X, keep_prob)
        X = tf.layers.conv2d_transpose(X, filters=64, kernel_size=4, strides=1, 
                             padding="same", activation=tf.nn.relu)
        X = tf.contrib.layers.flatten(X)
        X = tf.layers.dense(X, units=28*28, activation=tf.nn.sigmoid)
        img = tf.reshape(X, shape=[-1, 28, 28])
        
        return img

def main(): 

   # mnist, batch_size, X_in, Y_flat, keep_prob, number_of_latent_variables, reshaped_dim, inputs_decoder = prepare_data()
    sampled, mn, sd = encoder(X_in, keep_prob)
    dec = decoder(sampled, keep_prob)

    unreshaped = tf.reshape(dec, [-1, 28 * 28])
    img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1) #Log-likelihood 
    latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1) #Kullback-Leibler divergence
    loss = tf.reduce_mean(img_loss + latent_loss)
    optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    iterations = 1

    for i in range(iterations):
        
        batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
        #labels = mnist.train.labels
        labels = mnist.train.next_batch(batch_size=batch_size)[1]
        sess.run(optimizer, feed_dict={X_in: batch, Y: batch, keep_prob: 0.8})
        if not i % 200:
            ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict={X_in: batch, Y: batch, keep_prob: 1.0})
            plt.imshow(np.reshape(batch[0], [28, 28]), cmap="gray")
            #plt.show()
            plt.imshow(d[0], cmap="gray")
            #plt.show()
            if number_of_latent_variables == 20:
                coords = sess.run(sampled, feed_dict={X_in: batch, Y: batch, keep_prob: 0.8})
                colormap = ListedColormap(sns.color_palette(sns.hls_palette(10, l=.45, s=.8)).as_hex())
                pca = PCA(n_components=2)
                components = pca.fit_transform(coords)
                #plt.scatter(components[:,0], components[:,1], c=labels, cmap=colormap);
                #plt.colorbar()
                #plt.show()

    saver = tf.train.Saver()
    saver.save(sess, "./model.ckpt") 
    #Generate data
    print("Generating...")
    sess_load = tf.Session()
    randoms = [np.random.normal(0, 1, number_of_latent_variables) for _ in range(5)]
    imgs = sess.run(dec, feed_dict={sampled: randoms, keep_prob: 1.0})
    saver.restore(sess_load, "./model.ckpt" )
    variables = tf.get_collection(tf.GraphKeys.VARIABLES,
                               scope="encoder")
    print("a ", variables)
    sys.exit()
    imgs = sess.run(sess_load, feed_dict={sampled: randoms, keep_prob: 1.0})
    imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]
    
    for img in imgs:
        plt.figure(figsize=(1,1))
        plt.axis("off")
        plt.imshow(img, cmap="gray")
        plt.show()
        return plt

if __name__=="__main__":
    main()