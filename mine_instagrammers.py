from instagram_utils import Profile, analyzeInstagram
from keras.applications import ResNet50
import json

if __name__ == '__main__':
  model = ResNet50(weights='imagenet')

  todo = ['beauty_makeup', 'design', 'dogs', 'family', 'fashion', 'flower', 'food', 'healthy_living', 'illustration', 'lifestyle', 'music_artist', 'photographer', 'sports', 'travel', 'yoga']
  
  for to in todo:
    with open('./influencers/' + to + '.txt') as f:
      content = f.readlines()

    instagrammers = [x.strip() for x in content]

    tagFrequency = {}

    for i in instagrammers:
      print ('Tagging: ', i)
      try:
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
        with open('./data/' + to + '.json', 'w') as j:
            j.write(json.dumps(tagArr))

      except:
        print ('User does not exist')