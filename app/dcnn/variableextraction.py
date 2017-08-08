import tensorflow as tf


def setVariableMap():
    variable_dict = getVariables()
    for key in variable_dict.keys():
        if 'conv1' in key:
            tf.Variable(variable_dict[key], key.replace(':0', ''))
            print(key.replace(':0', ''))
        if 'conv2' in key:
            tf.Variable(variable_dict[key], key.replace(':0', ''))
            print(key.replace(':0', ''))

def getVariables():
    variable_dict = dict()
    with tf.Session() as sess:
      new_saver = tf.train.import_meta_graph('C:/tmp/cifar10_train/model.ckpt-55759.meta')
      new_saver.restore(sess, 'C:/tmp/cifar10_train/model.ckpt-55759')
      vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      for v in vars:
          print(v.name)
      #for v in vars:
       #   variable_dict[v.name] = sess.run(v)

    #return variable_dict