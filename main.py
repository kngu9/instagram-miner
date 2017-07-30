import json
import pandas as pd
import numpy as np
import urllib
from PIL import Image
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions


class Profile(object):
  def __init__(self, username):
    self.username = username
    self.user = json.loads(urllib.request.urlopen(f"https://www.instagram.com/{username}/?__a=1").read())['user']
    self.mediaIndex = None
    self.media = self.user['media']

  def nextMedia(self):
    if self.mediaIndex:
      self.media = json.loads(urllib.request.urlopen(f"https://www.instagram.com/{self.username}/?__a=1&max_id={self.mediaIndex}").read())['user']['media']

    self.mediaIndex = self.media['page_info']['end_cursor']

    return self.media['page_info']['has_next_page']

  def getMedia(self):
    return self.media['nodes']

def main():
  model = ResNet50(weights='imagenet')
  curProfile = Profile('dallas_foodie')

  while curProfile.nextMedia():  
    for m in curProfile.getMedia():
      if m['__typename'] == 'GraphImage':
        urllib.request.urlretrieve(m['display_src'], 'predict.jpg')
        img = image.load_img('predict.jpg', target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        print(f"Predicting: {m['display_src']}")
        print(f"Predicted: {decode_predictions(preds)}\n")

if __name__ == '__main__': 
  main()