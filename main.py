import os
import sys

import tensorflow as tf

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'util'))

from models import SegNet, ADDA
from utils import env_info

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('device', '0', 'Device to use [default: 0]')
tf.app.flags.DEFINE_string('mode', '', 'Program mode [train/test/finetune/da/datest default: None]')

tf.app.flags.DEFINE_string('log_dir', 'logs/', 'Directory to store ckpt and logs [default: logs/]')
tf.app.flags.DEFINE_string('ckpt', 'model-19999', 'Checkpoint file to load [default: model.ckpt-19999]')
tf.app.flags.DEFINE_string('train_path', 'dataset/train.txt', 'Train filename path [default: train.txt]')
tf.app.flags.DEFINE_string('test_path', 'dataset/test.txt', 'Test filename path [default: test.txt]')
tf.app.flags.DEFINE_string('val_path', 'dataset/val.txt', 'Validation filename path [default: val.txt]')
tf.app.flags.DEFINE_string('da_path', 'dataset/da.txt', 'Domain adaptation filename path [default: da.txt]')
tf.app.flags.DEFINE_boolean('output', True, ' Whether to save predicted image while testing [default: True]')

tf.app.flags.DEFINE_integer('image_h', '480', 'Image height [default: 480]')
tf.app.flags.DEFINE_integer('image_w', '480', 'Image width [default: 480]')
tf.app.flags.DEFINE_integer('image_c', '3', 'Image channel [default: 3]')
tf.app.flags.DEFINE_integer('num_classes', '3', 'Total class number [default: 3]')

tf.app.flags.DEFINE_integer('batch_size', '5', 'Batch size [default: 5]')
tf.app.flags.DEFINE_integer('max_steps', '20000', 'Total training steps [default: 20000]')
tf.app.flags.DEFINE_float('learning_rate', '1e-3', 'Initial learning rate [default: 1e-3]')


def main(args):
    env_info()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device)
    print('* ----------------------------------------------------------------- *')
    
    if FLAGS.mode == 'train':
        print('Program mode: train')
        print('Log directory: %s' % FLAGS.log_dir)
        print('Image size: %dx%d' % (FLAGS.image_h, FLAGS.image_w))
        print('Initial learning rate: %f' % FLAGS.learning_rate)
        print('Training steps: %d' % FLAGS.max_steps)
        print('* ----------------------------------------------------------------- *')
        segnet = SegNet(FLAGS)
        segnet.train()
    elif FLAGS.mode == 'test':
        print('Program mode: test')
        print('Log directory: %s' % FLAGS.log_dir)
        print('Image size: %dx%d' % (FLAGS.image_h, FLAGS.image_w))
        print('Name of heckpoint to restore: %s' % FLAGS.ckpt)
        print('* ----------------------------------------------------------------- *')
        segnet = SegNet(FLAGS)
        segnet.test()
    elif FLAGS.mode == 'finetune':
        print('Program mode: finetune')
        print('Log directory: %s' % FLAGS.log_dir)
        print('Image size: %dx%d' % (FLAGS.image_h, FLAGS.image_w))
        print('Learning rate: %f' % FLAGS.learning_rate)
        print('Name of heckpoint to restore: %s' % FLAGS.ckpt)
        print('Training steps: %d' % FLAGS.max_steps)
        print('* ----------------------------------------------------------------- *')
        segnet = SegNet(FLAGS)
        segnet.train()
    elif FLAGS.mode == 'da':
        print('Program mode: domain adaptation')
        print('Log directory: %s' % FLAGS.log_dir)
        print('Image size: %dx%d' % (FLAGS.image_h, FLAGS.image_w))
        print('Name of heckpoint to restore: %s' % FLAGS.ckpt)
        print('Training steps: %d' % FLAGS.max_steps)
        print('* ----------------------------------------------------------------- *')
        adda = ADDA(FLAGS)
        adda.train()
    elif FLAGS.mode == 'datest':
        print('Program mode: domain adaptation test')
        print('Image size: %dx%d' % (FLAGS.image_h, FLAGS.image_w))
        print('Name of heckpoint to restore: %s' % FLAGS.ckpt)
        print('* ----------------------------------------------------------------- *')
        adda = ADDA(FLAGS)
        adda.test()
    else:
        raise Exception('Invalid mode!')


if __name__ == '__main__':
    tf.app.run()
