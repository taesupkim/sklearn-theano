__author__ = 'KimTS'

import numpy
import glob
import cPickle as pkl
from sklearn_theano.datasets.base import load_resize_image
from sklearn_theano.feature_extraction.caffe import vgg
from scipy import sparse, vstack
from collections import OrderedDict

vgg_feature_extractor=vgg._get_fprop(output_layers=('pool5',), model=None, verbose=1)
batch_size = 100
resize = (448, 448)

def extract_coco_feature():
    image_dirs = ['/data/lisatmp4/taesup/data/coco/images/train2014',
                  '/data/lisatmp4/taesup/data/coco/images/val2014',
                  '/data/lisatmp4/taesup/data/coco/images/test2014',
                  '/data/lisatmp4/taesup/data/coco/images/test2015']
    desc_files  = ['/data/lisatmp4/taesup/data/coco/descriptors/train2014_desc.pkl',
                  '/data/lisatmp4/taesup/data/coco/descriptors/val2014_desc.pkl',
                  '/data/lisatmp4/taesup/data/coco/descriptors/test2014_desc.pkl',
                  '/data/lisatmp4/taesup/data/coco/descriptors/test2015_desc.pkl']
    for image_dir, desc_file in zip(image_dirs, desc_files):
        image_list = glob.glob(image_dir+'/*')
        feature_dict = OrderedDict()
        for image_path in image_list:
            image_data = load_resize_image(image_path)[None,]
            features = vgg_feature_extractor(image_data.transpose((0, 3, 1, 2)))[0][0]
            sparse_features = sparse.csr_matrix(features.reshape((features.shape[0],features.shape[1]*features.shape[2])))
            image_name = image_path.replace(image_dir,'')
            image_name = image_name.split('_')[2][:-4]
            image_name = int(image_name)

            feature_dict[image_name] = sparse_features
        with open(desc_file,'wb') as fp:
            pkl.dump(feature_dict,fp)

def extract_daquar_feature():
    image_dir = '/data/lisatmp4/taesup/data/daquar/image'
    train_list_filename = '/data/lisatmp4/taesup/data/daquar/train_image_list.txt'
    test_list_filename = '/data/lisatmp4/taesup/data/daquar/test_image_list.txt'
    train_desc_filename = '/data/lisatmp4/taesup/data/daquar/train_desc.pkl'
    test_desc_filename = '/data/lisatmp4/taesup/data/daquar/test_desc.pkl'

    with open(train_list_filename, 'r') as fp:
        train_images = fp.readlines()

    train_features = OrderedDict()
    for i, train_image  in enumerate(train_images):
        image_name = train_image.split()[0]
        image_path = image_dir + '/' + image_name + '.png'
        image_data = load_resize_image(image_path)[None,]
        features = vgg_feature_extractor(image_data.transpose((0, 3, 1, 2)))[0][0]
        sparse_features = sparse.csr_matrix(features.reshape((features.shape[0],features.shape[1]*features.shape[2])))
        train_features[image_name] = sparse_features
        print 'daquar train {}th image done'.format(i)

    with open(train_desc_filename,'wb') as fp:
        pkl.dump(train_features,fp)


    with open(test_list_filename, 'r') as fp:
        test_images = fp.readlines()

    test_features = OrderedDict()
    for i, test_image  in enumerate(test_images):
        image_name = test_image.split()[0]
        image_path = image_dir + '/' + image_name + '.png'
        image_data = load_resize_image(image_path)[None,]
        features = vgg_feature_extractor(image_data.transpose((0, 3, 1, 2)))[0][0]
        sparse_features = sparse.csr_matrix(features.reshape((features.shape[0],features.shape[1]*features.shape[2])))
        test_features[image_name] = sparse_features
        print 'daquar test {}th image done'.format(i)

    with open(test_desc_filename,'wb') as fp:
        pkl.dump(test_features,fp)

if __name__=="__main__":
    extract_daquar_feature()
    extract_coco_feature()