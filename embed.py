import sys
import os
from scipy import ndimage, misc
from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import pickle
import scipy.io
import pylab

model = {}
vgg_layers = None
NUM_CHANNELS = [64, 128, 256, 512, 512]
LAYER_IM_SIZE = [224, 112, 56, 28, 14]
EMBED_SIZE = sum(map(lambda x:x*x, NUM_CHANNELS))
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

def build_graph(graph):
    global model

    # extract weights and biases for a given convolutional layer
    def weights_and_biases(layer_index):
        W = tf.constant(vgg_layers[0][layer_index][0][0][2][0][0])
        b = vgg_layers[0][layer_index][0][0][2][0][1]
        b = tf.constant(np.reshape(b, (b.size))) # need to reshape b from size (64,1) to (64,)
        layer_name = vgg_layers[0][layer_index][0][0][0][0]
        return W,b

    with graph.as_default():
        model['image'] = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
        W,b = weights_and_biases(0)
        model['conv1_1'] = tf.nn.conv2d(model['image'], W, [1,1,1,1], 'SAME') + b
        model['relu1_1'] = tf.nn.relu(model['conv1_1'])
        W,b = weights_and_biases(2)
        model['conv1_2'] = tf.nn.conv2d(model['relu1_1'], W, [1,1,1,1], 'SAME') + b
        model['relu1_2'] = tf.nn.relu(model['conv1_2'])
        model['pool1'] = tf.nn.avg_pool(model['relu1_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        W,b = weights_and_biases(5)
        model['conv2_1'] = tf.nn.conv2d(model['pool1'], W, [1,1,1,1], 'SAME') + b
        model['relu2_1'] = tf.nn.relu(model['conv2_1'])
        W,b = weights_and_biases(7)
        model['conv2_2'] = tf.nn.conv2d(model['relu2_1'], W, [1,1,1,1], 'SAME') + b
        model['relu2_2'] = tf.nn.relu(model['conv2_2'])
        model['pool2'] = tf.nn.avg_pool(model['relu2_2'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        W,b = weights_and_biases(10)
        model['conv3_1'] = tf.nn.conv2d(model['pool2'], W, [1,1,1,1], 'SAME') + b
        model['relu3_1'] = tf.nn.relu(model['conv3_1'])
        W,b = weights_and_biases(12)
        model['conv3_2'] = tf.nn.conv2d(model['relu3_1'], W, [1,1,1,1], 'SAME') + b
        model['relu3_2'] = tf.nn.relu(model['conv3_2'])
        W,b = weights_and_biases(14)
        model['conv3_3'] = tf.nn.conv2d(model['relu3_2'], W, [1,1,1,1], 'SAME') + b
        model['relu3_3'] = tf.nn.relu(model['conv3_3'])
        W,b = weights_and_biases(16)
        model['conv3_4'] = tf.nn.conv2d(model['relu3_3'], W, [1,1,1,1], 'SAME') + b
        model['relu3_4'] = tf.nn.relu(model['conv3_4'])
        model['pool3'] = tf.nn.avg_pool(model['relu3_4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        W,b = weights_and_biases(19)
        model['conv4_1'] = tf.nn.conv2d(model['pool3'], W, [1,1,1,1], 'SAME') + b
        model['relu4_1'] = tf.nn.relu(model['conv4_1'])
        W,b = weights_and_biases(21)
        model['conv4_2'] = tf.nn.conv2d(model['relu4_1'], W, [1,1,1,1], 'SAME') + b
        model['relu4_2'] = tf.nn.relu(model['conv4_2'])
        W,b = weights_and_biases(23)
        model['conv4_3'] = tf.nn.conv2d(model['relu4_2'], W, [1,1,1,1], 'SAME') + b
        model['relu4_3'] = tf.nn.relu(model['conv4_3'])
        W,b = weights_and_biases(25)
        model['conv4_4'] = tf.nn.conv2d(model['relu4_3'], W, [1,1,1,1], 'SAME') + b
        model['relu4_4'] = tf.nn.relu(model['conv4_4'])
        model['pool4'] = tf.nn.avg_pool(model['relu4_4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        W,b = weights_and_biases(28)
        model['conv5_1'] = tf.nn.conv2d(model['pool4'], W, [1,1,1,1], 'SAME') + b
        model['relu5_1'] = tf.nn.relu(model['conv5_1'])
        W,b = weights_and_biases(30)
        model['conv5_2'] = tf.nn.conv2d(model['relu5_1'], W, [1,1,1,1], 'SAME') + b
        model['relu5_2'] = tf.nn.relu(model['conv5_2'])
        W,b = weights_and_biases(32)
        model['conv5_3'] = tf.nn.conv2d(model['relu5_2'], W, [1,1,1,1], 'SAME') + b
        model['relu5_3'] = tf.nn.relu(model['conv5_3'])
        W,b = weights_and_biases(34)
        model['conv5_4'] = tf.nn.conv2d(model['relu5_3'], W, [1,1,1,1], 'SAME') + b
        model['relu5_4'] = tf.nn.relu(model['conv5_4'])
        model['pool5'] = tf.nn.avg_pool(model['relu5_4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# read in image as array of pixels (RGB) and truncate to 224 x 224
def get_imarray(filename):
    array = ndimage.imread(filename, mode='RGB')
    array = np.asarray([misc.imresize(array, (224, 224))])
    return array

def gram_matrix(F, N, M):
    # F is the output of the given convolutional layer on a particular input image
    # N is number of feature maps in the layer
    # M is the total number of entries in each filter
    Ft = np.reshape(F, (M, N))
    return np.dot(np.transpose(Ft), Ft)

def flattened_gram(imarray, session):
    grams = np.empty([EMBED_SIZE])    
    index = 0
    for i in range(5):
        grams[index:(NUM_CHANNELS[i]**2 + index)] = gram_matrix(session.run(model['conv' + str(i+1) + '_1'], feed_dict={model['image']: imarray}), NUM_CHANNELS[i], LAYER_IM_SIZE[i]**2).flatten()
        index += NUM_CHANNELS[i]**2
    return grams

# distance between two style embeddings as defined in paper
def distance(fg1, fg2):
    dist = 0
    index = 0
    for i in range(5):
        square_1 = np.reshape(fg1[index:NUM_CHANNELS[i]**2 + index], (NUM_CHANNELS[i], NUM_CHANNELS[i]))
        square_2 = np.reshape(fg2[index:NUM_CHANNELS[i]**2 + index], (NUM_CHANNELS[i], NUM_CHANNELS[i]))
        index += NUM_CHANNELS[i]**2
        dist += (1.0 / (4 * NUM_CHANNELS[i] * LAYER_IM_SIZE[i]**2)) * (np.linalg.matrix_power(square_1 - square_2, 2)).sum()
    return dist

def main(argv):
    global vgg_layers
    usage = "usage: python embed.py [image_directory] [output_directory]"
    im_dir = None
    output_dir = None
    # parse command line arguments
    if len(argv) != 2:
        print(usage)
        sys.exit()
    if os.path.isdir(argv[0]) == False:
        print(argv[0] + " is not a valid directory")
        sys.exit()
    else:
        im_dir = argv[0]
    if os.path.isdir(argv[1]) == False:
        print(argv[1] + " is not a valid directory")
        sys.exit()
    else:
        output_dir = argv[1]

    # load VGG model matrix from file
    vgg = scipy.io.loadmat(VGG_MODEL)
    vgg_layers = vgg['layers']
    print("VGG matrix loaded...")

    # build the tensorflow graph
    graph = tf.Graph()
    build_graph(graph)
    print("Tensorflow graph built...")
    del vgg # to free up memory

    filenames = []
    for filename in os.listdir(im_dir):
        if os.path.splitext(filename)[1] in ('.jpg', '.png'):
            filenames.append(os.path.split(filename)[1])

    embeddings = np.empty([len(filenames), EMBED_SIZE])

    with tf.Session(graph=graph) as sess:
        count = 0
        for filename in filenames:
            embeddings[count, :] = flattened_gram(get_imarray(os.path.join(im_dir ,filename)), sess)
            count += 1
            if count % 10 == 0:
                print("Embedded " + str(count) + " images")
            
    print("Large embeddings generated. Shape: " + str(embeddings.shape))

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, metric=distance)
    print("Projecting onto two dimensions... this might take a while")
    two_d_embeddings = tsne.fit_transform(embeddings)
    print("2D embeddings generated")

    plot_data = {'embeddings': two_d_embeddings, 'filenames': filenames}
    pickle.dump( plot_data, open( os.path.join(output_dir, os.path.split(im_dir)[1]) + '_embed.pickle', "wb" ) )
    print("Pickle dumped")

    # def plot(embeddings):
    #   fig = pylab.figure(figsize=(15,15))  # in inches
    #   x = embeddings[:, 0]
    #   y = embeddings[:, 1]
    #   def onpick3(event):
    #     ind = event.ind[0]
    #     print(filenames[ind], np.take(x, ind), np.take(y, ind))
    #   pylab.scatter(x, y, picker=True)
    #   fig.canvas.mpl_connect('pick_event', onpick3)
    #   # pylab.colorbar()
    #   pylab.show()


    # plot(two_d_embeddings)

if __name__ == "__main__":
    main(sys.argv[1:])



