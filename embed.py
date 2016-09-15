
# coding: utf-8

# In[1]:

import os
from IPython.display import Image
from scipy import ndimage, misc
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pylab
import pickle

import scipy.io
from scipy.stats import norm
# Config the matlotlib backend as plotting inline in IPython
get_ipython().magic(u'matplotlib inline')


# In[2]:

VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'


# In[3]:

vgg = scipy.io.loadmat(VGG_MODEL)
vgg_layers = vgg['layers']


# In[4]:

def weights_and_biases(layer_index):
    W = tf.constant(vgg_layers[0][layer_index][0][0][2][0][0])
    b = vgg_layers[0][layer_index][0][0][2][0][1]
    b = tf.constant(np.reshape(b, (b.size))) # need to reshape b from size (64,1) to (64,)
    layer_name = vgg_layers[0][layer_index][0][0][0][0]
    return W,b


# In[5]:

image_size = 224

graph = tf.Graph()

with graph.as_default():
    tf_image = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))

    output = {}
    W,b = weights_and_biases(0)
    output['conv1_1'] = tf.nn.conv2d(tf_image, W, [1,1,1,1], 'SAME') + b
    output['relu1_1'] = tf.nn.relu(output['conv1_1'])
    W,b = weights_and_biases(2)
    output['conv1_2'] = tf.nn.conv2d(output['relu1_1'], W, [1,1,1,1], 'SAME') + b
    output['relu1_2'] = tf.nn.relu(output['conv1_2'])
    output['pool1'] = tf.nn.avg_pool(output['relu1_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    W,b = weights_and_biases(5)
    output['conv2_1'] = tf.nn.conv2d(output['pool1'], W, [1,1,1,1], 'SAME') + b
    output['relu2_1'] = tf.nn.relu(output['conv2_1'])
    W,b = weights_and_biases(7)
    output['conv2_2'] = tf.nn.conv2d(output['relu2_1'], W, [1,1,1,1], 'SAME') + b
    output['relu2_2'] = tf.nn.relu(output['conv2_2'])
    output['pool2'] = tf.nn.avg_pool(output['relu2_2'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    W,b = weights_and_biases(10)
    output['conv3_1'] = tf.nn.conv2d(output['pool2'], W, [1,1,1,1], 'SAME') + b
    output['relu3_1'] = tf.nn.relu(output['conv3_1'])
    W,b = weights_and_biases(12)
    output['conv3_2'] = tf.nn.conv2d(output['relu3_1'], W, [1,1,1,1], 'SAME') + b
    output['relu3_2'] = tf.nn.relu(output['conv3_2'])
    W,b = weights_and_biases(14)
    output['conv3_3'] = tf.nn.conv2d(output['relu3_2'], W, [1,1,1,1], 'SAME') + b
    output['relu3_3'] = tf.nn.relu(output['conv3_3'])
    W,b = weights_and_biases(16)
    output['conv3_4'] = tf.nn.conv2d(output['relu3_3'], W, [1,1,1,1], 'SAME') + b
    output['relu3_4'] = tf.nn.relu(output['conv3_4'])
    output['pool3'] = tf.nn.avg_pool(output['relu3_4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    W,b = weights_and_biases(19)
    output['conv4_1'] = tf.nn.conv2d(output['pool3'], W, [1,1,1,1], 'SAME') + b
    output['relu4_1'] = tf.nn.relu(output['conv4_1'])
    W,b = weights_and_biases(21)
    output['conv4_2'] = tf.nn.conv2d(output['relu4_1'], W, [1,1,1,1], 'SAME') + b
    output['relu4_2'] = tf.nn.relu(output['conv4_2'])
    W,b = weights_and_biases(23)
    output['conv4_3'] = tf.nn.conv2d(output['relu4_2'], W, [1,1,1,1], 'SAME') + b
    output['relu4_3'] = tf.nn.relu(output['conv4_3'])
    W,b = weights_and_biases(25)
    output['conv4_4'] = tf.nn.conv2d(output['relu4_3'], W, [1,1,1,1], 'SAME') + b
    output['relu4_4'] = tf.nn.relu(output['conv4_4'])
    output['pool4'] = tf.nn.avg_pool(output['relu4_4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    W,b = weights_and_biases(28)
    output['conv5_1'] = tf.nn.conv2d(output['pool4'], W, [1,1,1,1], 'SAME') + b
    output['relu5_1'] = tf.nn.relu(output['conv5_1'])
    W,b = weights_and_biases(30)
    output['conv5_2'] = tf.nn.conv2d(output['relu5_1'], W, [1,1,1,1], 'SAME') + b
    output['relu5_2'] = tf.nn.relu(output['conv5_2'])
    W,b = weights_and_biases(32)
    output['conv5_3'] = tf.nn.conv2d(output['relu5_2'], W, [1,1,1,1], 'SAME') + b
    output['relu5_3'] = tf.nn.relu(output['conv5_3'])
    W,b = weights_and_biases(34)
    output['conv5_4'] = tf.nn.conv2d(output['relu5_3'], W, [1,1,1,1], 'SAME') + b
    output['relu5_4'] = tf.nn.relu(output['conv5_4'])
    output['pool5'] = tf.nn.avg_pool(output['relu5_4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def gram_matrix(F, N, M):
    # F is the output of the given convolutional layer on a particular input image
    # N is number of feature maps in the layer
    # M is the total number of entries in each filter
    Ft = np.reshape(F, (M, N))
    return np.dot(np.transpose(Ft), Ft)


# Build the dataset of Gram matrices

# In[6]:

def get_imarray(filename):
    array = ndimage.imread('picasso/' + filename, mode='RGB')
    array = np.asarray([misc.imresize(array, (224, 224))])
    return array


# In[7]:

conv_depth = 5
num_layers = [64, 128, 256, 512, 512]
layer_size = [224, 112, 56, 28, 14]

embed_size = 0
for i in range(conv_depth):
    embed_size += num_layers[i]**2   


# In[8]:

print(embed_size)


# In[9]:

def flattened_gram(imarray, session):
    grams = np.empty([embed_size])    
    index = 0
    for i in range(conv_depth):
        grams[index:(num_layers[i]**2 + index)] = gram_matrix(session.run(output['conv' + str(i+1) + '_1'], feed_dict={tf_image: imarray}), num_layers[i], layer_size[i]**2).flatten()
        index += num_layers[i]**2
    return grams


# In[10]:

max_images = 359
filenames = []
year_dict = {}
min_year = 3000
max_year = 0

embeddings = np.empty([max_images, embed_size])

with tf.Session(graph=graph) as sess:
    count = 0
    with open('picasso.csv') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            year = row['date']
            if year == '':
                continue
            filename = row['filename']
            filenames.append(filename)
            embeddings[count, :] = flattened_gram(get_imarray(filename), sess)
            year = int(year.split('.')[-1])
            if year < min_year:
                min_year = year
            if year > max_year:
                max_year = year
            year_dict[row['filename']] = year
            count += 1
            if count >= max_images:
                break
        
print(embeddings.shape)
    


# In[11]:

def distance(fg1, fg2):
    dist = 0
    index = 0
    for i in range(conv_depth):
        square_1 = np.reshape(fg1[index:num_layers[i]**2 + index], (num_layers[i], num_layers[i]))
        square_2 = np.reshape(fg2[index:num_layers[i]**2 + index], (num_layers[i], num_layers[i]))
        index += num_layers[i]**2
        dist += (1.0 / (4 * num_layers[i] * layer_size[i]**2)) * (np.linalg.matrix_power(square_1 - square_2, 2)).sum()
    return dist


# In[12]:

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, metric=distance)
two_d_embeddings = tsne.fit_transform(embeddings)


# In[13]:

year_labels = []

for filename in filenames:
    year_labels.append(year_dict[filename])
    #if year_dict[filename] == 0:
        #year_labels.append(0)
    #else:
        #year_labels.append((year_dict[filename] - min_year) / float(max_year-min_year))


# In[14]:

ids = [filename.split('.')[0] for filename in filenames]


# In[15]:

plot_data = {'embeddings': two_d_embeddings, 'years': year_labels, 'ids': ids}
pickle.dump( plot_data, open( "plot_data.pickle", "wb" ) )


# In[16]:

def plot(embeddings, years, labels):
  assert embeddings.shape[0] >= len(year_labels), 'More labels than embeddings'
  fig = pylab.figure(figsize=(15,15))  # in inches
  x = embeddings[:, 0]
  y = embeddings[:, 1]
  def onpick3(event):
    ind = event.ind
    print 'onpick3 scatter:', ind, np.take(x, ind), np.take(y, ind)
  pylab.scatter(x, y, c=years, picker=True)
  fig.canvas.mpl_connect('pick_event', onpick3)
  pylab.colorbar()
  pylab.show()


plot(two_d_embeddings, year_labels, ids)


# In[ ]:



