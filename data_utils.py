from PIL import Image,ImageOps,ImageDraw
import os,random,pickle
import numpy as np


# ALERT ------changing these values will affect directory organization in output
origianl_image_path = '/home/mukherjee/Work/Handwriting/Debnagari/HindiOp2_1'
cropped_image_path = './Data/images'
height_norm_image_path = 'Data/HN/'
# ALERT ------changing these values will affect directory organization in output


def get_wh_info(image_dir):
    max_w = max_h = 0
    min_w = min_h = 10000
    for root,sd,files in os.walk(image_dir):
        for filename in files:
            if(filename[-3:]=='jpg'):
                abspath = os.path.join(root,filename)
                img = Image.open(abspath)
                w,h = img.size
                if(w>max_w):
                    max_w = w
                if(w<min_w):
                    min_w = w
                if(h>max_h):
                    max_h = h
                if(h<min_h):
                    min_h = h
    print("Max Width %d Max Height %d"%(max_w,max_h))
    print("Min Width %d Min Height %d" % (min_w, min_h))

# crop images as per boundary box
def crop_image(image_file,save_in):
    path_info = image_file.split('/')
    target_dir = save_in + path_info[-3] + "/" + path_info[-2]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print('created ',target_dir)
    else:
        print('existing ',target_dir)
    im = Image.open(image_file)
    im_i = ImageOps.invert(im)
    left, upper, right, lower = im_i.getbbox()
    tempImage = im_i.crop((left, upper, right, lower))
    newPath = os.path.join(target_dir,path_info[-1])
    tempImage.save(newPath)
    print('%s cropped and saved as %s'%(image_file,newPath))

def resize_image(image_file,save_in,maxh):
    path_info = image_file.split('/')
    target_dir = save_in + path_info[-3] + "/" + path_info[-2]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print('created ',target_dir)
    else:
        print('existing ',target_dir)
    im = Image.open(image_file)
    w,h = im.size
    ar = w / h
    new_h = maxh
    new_w = int(new_h*ar)
    tempImage = im.resize((new_w,new_h))
    newPath = os.path.join(target_dir,path_info[-1])
    tempImage.save(newPath)
    print('%s resized and saved as %s'%(image_file,newPath))

# crop/resize all images in a directory
def apply_on_all_images(original_images_dir,save_to,task='resize',maxh=None):
    for root,sd,files in os.walk(original_images_dir):
        for filename in files:
            abspath = os.path.join(root,filename)
            if(task=='crop'):
                crop_image(abspath,save_to)
            else:
                resize_image(abspath,save_to,maxh=maxh)

# creating a list of all files in dataset along with width, height and groundtruth
# also create a list of unique characters
# also create a list of unique words
def gather_image_info(image_dir,gtfile,out_file):
    # make a gt dict
    gt_dict = {}
    f = open(gtfile)
    line = f.readline()
    while line:
        info = line.strip('\n').split(',')
        path_info = info[0].split('/')
        image_path = path_info[1] + "/" + path_info[2] +"/" + path_info[3]
        gt_dict[image_path] = info[-1]
        line = f.readline()
    f.close()
    # print(gt_dict)
    # now we have a gt dict
    # create a list of image_file,width,height,groundtruth
    f = open(out_file,'w')
    f_miss = open('missing_data.txt','w')
    all_words = []
    all_chars = []
    for root,sd,files in os.walk(image_dir):
        for filename in files:
            abs_path = os.path.join(root,filename)
            # print(abs_path.split('/'))
            rel_path = abs_path.split('/')[2] + "/" +abs_path.split('/')[3] + "/" + filename # key of gt
            try:
                this_gt = gt_dict[rel_path].replace('-','')
                unicode_gt = this_gt.encode('unicode-escape').decode('utf-8').replace('\\',' ')
                chars = unicode_gt.split()
                all_chars.extend(chars)
                all_words.append(this_gt)
                im = Image.open(abs_path)
                width, height = im.size
                info = "%s,%s,%d,%d,%s,%s\n"%(abs_path,rel_path,width,height,this_gt,unicode_gt)
                f.write(info)
            except:
                f_miss.write(abs_path+"\n")
                pass
    f.close()
    f_miss.close()
    all_chars = list(set(all_chars))
    all_words = list(set(all_words))
    f = open('Data/characters.txt','w')
    f.write("<BLANK>\n")
    for ch in all_chars:
        ch = '\\'+ch
        f.write("%s\n"%(ch))
    f.write("<PAD>\n")
    f.close()
    f = open('Data/words.txt','w')
    for wd in all_words:
        f.write("%s\n"%wd)
    f.close()

def load_tokens(token_file):
    tok2ind = {}
    ind2tok = {}
    f = open(token_file)
    line = f.readline()
    i = 0
    while line:
        info = line.strip('\n')
        token = info.replace('\\','')
        tok2ind[token] = i
        ind2tok[i] = token
        i += 1
        line = f.readline()
    f.close()
    return tok2ind,ind2tok

# create two text files containing list of train and test data
def create_random_train_test(all_files,train_file,test_file,split=0.3):
    all_data = []
    f = open(all_files)
    line = f.readline()
    while line:
        all_data.append(line)
        line = f.readline()
    print('gathered all file information')
    random.shuffle(all_data)
    print('shuffled')
    total = len(all_data)
    nb_train = int(total*(1-split))
    train_data = all_data[:nb_train]
    test_data = all_data[nb_train:]
    f = open(train_file,'w')
    for d in train_data:
        f.write(d)
    f.close()
    print('train file ready')
    f = open(test_file, 'w')
    for d in test_data:
        f.write(d)
    f.close()
    print('test file ready')

# create a pkls for list of images , dict with relpath as key
def dump_all_images(list_of_images_file,img_pkl_name):
    all_images = {}
    f = open(list_of_images_file)
    line = f.readline()
    while line:
        info = line.strip('\n').split(",")
        file_name = info[1] # load relative path
        file_path = info[0]
        img = Image.open(file_path)
        pix = img.getdata()
        pix = np.reshape(pix,[img.size[1],img.size[0]]).astype('int8')
        print('image size ',pix.shape)
        all_images[file_name] = pix
        line = f.readline()
    with open(img_pkl_name,'wb') as pkl:
        pickle.dump(all_images,pkl)
    pkl.close()

def get_congig(config_file):
    config = {}
    f = open(config_file)
    line = f.readline()
    while line:
        info = line.strip("\n").split(":")
        key = info[0]
        values = [v for v in info[1].split()]
        if(len(values)==1):
            config[key] = values[0]
        else:
            config[key] = values
        line = f.readline()
    f.close()
    return config

def initiate_batch_generation(pkl_img_file,list_file):
    with open(pkl_img_file,'rb') as f:
        all_image_array = pickle.load(f)
    f = open(list_file)
    lines = f.readlines()
    total = len(lines)
    return all_image_array,total

def batch_generator(list_images,batch_size,pkl_img,token_file):
    tok2ind,ind2tok = load_tokens(token_file)
    f = open(list_images)
    lines = f.readlines()
    total = len(lines)
    start = 0
    while True:
        end = min(total, start + batch_size)
        batch_list = lines[start:end]
        # find max_w from batch
        all_im_files = []
        all_ws = []
        all_hs = []
        all_gt_lens = []
        all_gts = []
        for b in batch_list:
            info = b.strip('\n').split(',')
            all_im_files.append(info[1])
            gt_split = info[-1].split()
            all_gts.append(gt_split)
            all_gt_lens.append(len(gt_split))
            all_hs.append(int(info[3]))
            all_ws.append(int(info[2]))
        max_w = max(all_ws)
        max_h = max(all_hs)
        max_y_len = max(all_gt_lens)
        bx = []
        by = []
        bx_lens = []
        by_lens = []
        # collect image arrays and normalize ,
        # they have same height so pad to max width of batch
        for k in range(len(all_im_files)):
            pix = pkl_img[all_im_files[k]]
            tempimage = Image.fromarray(pix.astype('uint8'),'L')
            canvas = Image.new('L',(max_w,max_h))
            canvas.paste(tempimage)
            pix = canvas.getdata()
            pix = np.reshape(pix, [canvas.size[1], canvas.size[0]]).astype('uint8')/255.0
            bx.append(pix)
            bx_lens.append(all_ws[k])
            canvas.save('temp.jpg')
            gt_ind = [tok2ind['<PAD>'] for _ in range(max_y_len)]
            i = 0
            y_length = len(all_gts[k])
            for j in range(y_length):
                ind = tok2ind[all_gts[k][j]]
                gt_ind[i] = ind
                i += 1
            by.append(gt_ind)
            by_lens.append(y_length)
        bx = np.asarray(bx)
        by = np.asarray(by)
        yield bx,bx_lens,by,by_lens,start,end
        start = end
        if(start >= total):
            start = 0

def get_all_image_keys(all_img_file):
    f = open(all_img_file)
    all_keys = []
    all_gts = []
    line = f.readline()
    while line:
        info = line.strip('\n').split(',')
        key = info[1]
        gt = info[-1]
        all_keys.append(key)
        all_gts.append(gt)
        line = f.readline()
    f.close()
    return all_keys,all_gts

# apply_on_all_images(origianl_image_path,cropped_image_path,task='crop')
# get_wh_info(cropped_image_path)
# apply_on_all_images(cropped_image_path,height_norm_image_path,maxh=36)
# gather_image_info('Data/HN','Data/groundtruth.txt','all_files.txt')
# create_random_train_test('all_files.txt','train_1.txt','test_1.txt',split=0.2)
# dump_all_images('all_files.txt','all_images.pkl')
# how to generate batches

# pkl_img,total = initiate_batch_generation('Data/all_images.pkl','Data/train_1.txt')
# batches = batch_generator('Data/train_1.txt',32,pkl_img,'Data/characters.txt')
# nb_batches = int(np.ceil(total/32))
# for b in range(nb_batches):
#     bx,bx_lens,by,by_lens,start,end = next(batches)
#     print(b,bx.shape,by.shape,start,end)