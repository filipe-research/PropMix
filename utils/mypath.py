import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'cifar-100', 'animal10n', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200'}
        assert(database in db_names)

        if database == 'cifar-10':
            return '../../data/cifar10/'
        
        if database == 'cifar-100':
            return '../../data/cifar100'
        
        else:
            raise NotImplementedError
