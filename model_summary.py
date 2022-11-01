from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
args = vars(ap.parse_args())


model = load_model(args["model"])

print(model.summary())
print (' Input names :',model.inputs)
print (' Output names:', model.outputs)