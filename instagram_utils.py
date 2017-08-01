import urllib
import json
import numpy as np
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import requests

class Profile(object):
  def __init__(self, username):
    self.username = username
    self.user = json.loads(requests.get('https://www.instagram.com/' + username + '/?__a=1').text)['user']
    self.mediaIndex = None
    self.media = self.user['media']

  def nextMedia(self):
    if self.mediaIndex:
      self.media = json.loads(requests.get('https://www.instagram.com/' + self.username + '/?__a=1&max_id=' + self.mediaIndex).text)['user']['media']

    self.mediaIndex = self.media['page_info']['end_cursor']

    return self.media['page_info']['has_next_page']

  def getMedia(self):
    return self.media['nodes']

def analyzeInstagram(model, username):
  curProfile = Profile(username)

  tagFrequency = {}

  while curProfile.nextMedia():
    for m in curProfile.getMedia():
      if m['__typename'] == 'GraphImage':
        urllib.request.urlretrieve(m['display_src'], 'predict.jpg')
        img = image.load_img('predict.jpg', target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = decode_predictions(model.predict(x))

        for p in preds:
          for tag in p:
            if tag[1] not in tagFrequency:
              tagFrequency[tag[1]] = 0
            tagFrequency[tag[1]] += 1
    print(tagFrequency)

  return tagFrequency
  
