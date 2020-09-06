from model import *
from data_utils import *
pkl_path = '/home/mukherjee/Work/Handwriting/Data/all_images.pkl'
train_file = '/home/mukherjee/Work/Handwriting/Data/train_1.txt'
test_file = '/home/mukherjee/Work/Handwriting/Data/test_1.txt'
tokfile = '/home/mukherjee/Work/Handwriting/Data/characters.txt'
testnet = HandWritingRecognizer('config')
# testnet.train(pkl_path,train_file,test_file,tokfile)

test_images,total  = initiate_batch_generation(pkl_path,test_file)
print(len(test_images))

all_keys,all_gts = get_all_image_keys(test_file)
print(len(all_keys))

image_key = all_keys[15000]
grounttruth = all_gts[15000]
random_image = test_images[image_key]

print(random_image.shape)

tok2ind,ind2tok = load_tokens(tokfile)
prediction_ind,prediction_tok = testnet.predict(random_image,random_image.shape[1],ind2tok)
print(prediction_tok)
print(grounttruth)

