from keras.applications import ResNet50
from instagram_utils import Profile, analyzeInstagram
from tf_idf import createTFIDFMatrix
from train import buildModel


def main():
  buildModel()
  model = ResNet50(weights='imagenet')  
  tags = analyzeInstagram(model, 'clemfoodie')

  createTFIDFMatrix(tags)

if __name__ == '__main__': 
  main()