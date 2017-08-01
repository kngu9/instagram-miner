from instagram_utils import Profile, analyzeInstagram
from keras.applications import ResNet50
import json

if __name__ == '__main__':
  model = ResNet50(weights='imagenet')

  with open('./influencers/aviation.txt') as f:
    content = f.readlines()

  instagrammers = [x.strip() for x in content]

  tagFrequency = {}

  for i in instagrammers:
    print ('Tagging: ', i)

    tags = analyzeInstagram(model, i)

    for t in tags:
      if t['tag'] not in tagFrequency:
        tagFrequency[t['tag']] = {'count': 0, 'weightSum': 0}

      tagFrequency[t['tag']]['count'] += 1
      tagFrequency[t['tag']]['weightSum'] += t['weight']
  
  tagArr = []

  for t in tagFrequency:
    curObj = tagFrequency[t]
    tagArr.append({'tag': t, 'weight': curObj['count']/(curObj['weightSum']/curObj['count'])})
  
  tagArr = sorted(tagArr, key=lambda k: k['weight'], reverse=False)
  with open('./data/aviation.json', 'w') as j:
      j.write(json.dumps(tagArr))