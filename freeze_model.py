import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
args = vars(ap.parse_args())

# set learning phase for no training
keras.backend.set_learning_phase(0)

# load weights & architecture into new model
loaded_model = load_model(args["model"])

tf_ckpt = 'tfchkpt.ckpt'
tf_session = tf.compat.v1.Session()
saver = tf.compat.v1.train.Saver()
save_path = saver.save(tf_session, tf_ckpt)

print(save_path)