import tensorflow as tf
from tensorflow.python.platform import gfile


class TensorflowModel(object):

    def __init__(self, modelPath):
        self.model = self.loadModel(modelPath)

    def loadModel(self, modelPath):
        # Load the saved graph
        with gfile.GFile(modelPath, 'rb') as f:
            graph_def = tf.GraphDef()
            try:
                graph_def.ParseFromString(f.read())
            except:
                print('try adding PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' +
                      'to environment.  e.g.:\n' +
                      'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ipython\n' +
                      'See here for info: ' +
                      'https://github.com/tf/tf/issues/582')
            return graph_def
