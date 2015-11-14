__author__ = 'KimTS'

from sklearn_theano.feature_extraction.caffe import vgg


if __name__=="__main__":
    vgg_feature_extractor=vgg._get_fprop(output_layers=('prob',), model=None, verbose=0)

    print 'done'