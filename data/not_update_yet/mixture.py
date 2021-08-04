import os,sys
import os.path
import numpy as np
import torch
import torch.utils.data
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets,transforms
import urllib.request
from PIL import Image
import pickle

#from __future__ import print_function
import warnings
import codecs

#from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, \
    makedir_exist_ok, verify_str_arg, check_integrity

from torchvision.datasets import CIFAR10, FashionMNIST, MNIST, omniglot


#####################################################################

class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)

##############################################################################
###################################################


class SVHN(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root, split='train', transform=None, target_transform=None,
                 download=False):
        super(SVHN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.targets = self.labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)





###############################################################################


class MY_MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MY_MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)   #这里就是tensor了

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = torch.cat([img, img, img], axis=0)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class MY_FashionMNIST(MY_MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Fashion-MNIST/processed/training.pt``
            and  ``Fashion-MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    resources = [
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
         "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
         "25c81989df183df01b3e8a0aad5dffbe"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
         "bef4ecab320f06d8554ea6380940ec79"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
         "bb300cfdad3c16e7a12a480ee83cd310")
    ]
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')}
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x
##########################################################################################################
def get_dataset(bsz, num_task):
    #tasks = [MY_FashionMNIST, CIFAR10, TrafficSigns, Facescrub, notMNIST, MY_MNIST]   #0,4,5
    #tasks = [CIFAR10, MY_FashionMNIST, Facescrub, notMNIST, MY_MNIST]
    tasks = [SVHN, CIFAR10, MY_FashionMNIST, notMNIST, MY_MNIST]

    #tasks = [TrafficSigns, Facescrub, notMNIST, MY_MNIST, MY_FashionMNIST, CIFAR10]   #0,4,5    

    # task = [CIFAR10, SVHN, FashionMNIST, TrafficSigns, Facescrub, notMNIST, MNIST, omniglot]

    cifar_transform  = transforms.Compose([transforms.ToTensor(),transforms.Normalize([x/255 for x in [125.3,123.0,113.9]],[x/255 for x in [63.0,62.1,66.7]])])
    svhn_transform   = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.4377,0.4438,0.4728], [0.198,0.201,0.197])])
    fmnist_transform = transforms.Compose([transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize((0.2190,), (0.3318,))])
    tsign_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.3398,0.3117,0.3210], [0.2755,0.2647,0.2712])])
    fscrub_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5163,0.5569,0.4695], [0.2307,0.2272,0.2479])])
    nmnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4254,), (0.4501,))])
    mnist_transform  = transforms.Compose([transforms.Pad(padding=2,fill=0), transforms.ToTensor(), transforms.Normalize((0.1,), (0.2752,))])
    # omnig_transform  = transforms.Compose([transforms.Resize((28, 28)), transforms.Pad(padding=2, fill=0), transforms.ToTensor()])

    #transform = [fmnist_transform, cifar_transform, tsign_transform, fscrub_transform, nmnist_transform, mnist_transform]
    #transform = [cifar_transform, fmnist_transform, fscrub_transform, nmnist_transform, mnist_transform]
    transform = [svhn_transform, cifar_transform, fmnist_transform, nmnist_transform, mnist_transform]
    #transform = [tsign_transform, fscrub_transform, nmnist_transform, mnist_transform, fmnist_transform, cifar_transform]


    train_loader, test_loader, labels, tot_num_labels = {}, {}, {}, 0
    
    for idx, dataset in enumerate(tasks):
        # if idx == 1:
        #     trainset = dataset(root='./dataset/mixture', split="train", download=True, transform=transform[idx])
        #     trainset = dataset(root='./dataset/mixture', split="test", download=True, transform=transform[idx])
        # else:
        if idx == 0:
            trainset = dataset(root='./dataset/mixture', split='train', download=True, transform=transform[idx])
            testset = dataset(root='./dataset/mixture', split='test', download=True, transform=transform[idx])
        else:
            trainset = dataset(root='./dataset/mixture', train=True, download=True, transform=transform[idx])
            testset = dataset(root='./dataset/mixture', train=False, download=True, transform=transform[idx])
        
        for i, target in enumerate(trainset.targets):
            trainset.targets[i] = int(target + tot_num_labels)
        for i, target in enumerate(testset.targets):
            testset.targets[i] = int(target + tot_num_labels)

        # if (idx == 0) or (idx == 4) or (idx == 5):
        #     for i,train_data in enumerate(trainset.data):

        #         trainset.data[i] = torch.stack((train_data, train_data, train_data), axis=0)
        #         #trainset.data[i]=train_data.expand(1,3,train_data.size(2),train_data.size(3)) # Create 3 equal channels    
        #     for i,test_data in enumerate(testset.data):

        #         testset.data[i] = torch.stack((test_data, test_data, test_data), axis=0)                
        #         #testset.data[i]=test_data.expand(1,3,test_data.size(2),test_data.size(3)) # Create 3 equal channels  

        train_loader[idx] = DataLoader(trainset, batch_size=bsz, shuffle=False)
        test_loader[idx] = DataLoader(testset, batch_size=bsz, shuffle=False)
      
        labels[idx] = trainset.targets.unique() if isinstance(trainset.targets, torch.Tensor) else list(set(trainset.targets))

        # print(idx, labels[idx], len(labels[idx]))
        tot_num_labels += len(labels[idx])
    
    print(labels.keys())
    return train_loader, test_loader, labels


########################################################################################################################

class TrafficSigns(torch.utils.data.Dataset):
    """`German Traffic Signs <http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self, root, train=True,transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = "traffic_signs_dataset.zip"
        self.url = "https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip"
        # Other options for the same 32x32 pickled dataset
        # url="https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip"
        # url_train="https://drive.google.com/open?id=0B5WIzrIVeL0WR1dsTC1FdWEtWFE"
        # url_test="https://drive.google.com/open?id=0B5WIzrIVeL0WLTlPNlR2RG95S3c"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                self.download()

        training_file = 'lab 2 data/train.p'
        testing_file = 'lab 2 data/test.p'
        if train:
            with open(os.path.join(root,training_file), mode='rb') as f:
                train = pickle.load(f)
            self.data = train['features']
            self.labels = train['labels']
        else:
            with open(os.path.join(root,testing_file), mode='rb') as f:
                test = pickle.load(f)
            self.data = test['features']
            self.labels = test['labels']
        
        label_idx_under10 = []
        for idx, label in enumerate(self.labels):
            if int(self.labels[idx]) < 10:
                label_idx_under10.append(idx)
        
        print('self.data.shape:', self.data.shape, 'self.labels.shape:', self.labels.shape)

        self_data_tmp = self.data[label_idx_under10]
        self_labels_tmp = self.labels[label_idx_under10]

        print('self_data_tmp.shape:', self_data_tmp.shape, 'self_labels_tmp.shape:', self_labels_tmp.shape)

        #self.data = np.transpose(self_data_tmp, (0, 3, 1, 2))
        self.data = self_data_tmp      
        self.labels = self_labels_tmp.astype(np.int64)
        self.targets = self.labels

        print('self.data.shape:', self.data.shape, 'self.labels.shape:', self.labels.shape)        

        #self.data = np.transpose(self.data, (0, 3, 1, 2))
        #self.targets = self.labels
        #print(self.data.shape); sys.exit()

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)
        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)
        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()


########################################################################################################################

class Facescrub(torch.utils.data.Dataset):
    """Subset of the Facescrub cropped from the official Megaface challenge page: http://megaface.cs.washington.edu/participate/challenge.html, resized to 38x38
    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self, root, train=True,transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = "facescrub_100.zip"
        self.url = "https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_100.zip?raw=true"

        fpath=os.path.join(root,self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                self.download()

        training_file = 'facescrub_train_100.pkl'
        testing_file = 'facescrub_test_100.pkl'
        if train:
            with open(os.path.join(root,training_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # train  = u.load()
                train = pickle.load(f)
            self.data = train['features'].astype(np.uint8)
            self.labels = train['labels'].astype(np.uint8)
            """
            print(self.data.shape)
            print(self.data.mean())
            print(self.data.std())
            print(self.labels.max())
            #"""
        else:
            with open(os.path.join(root,testing_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # test  = u.load()
                test = pickle.load(f)

            self.data = test['features'].astype(np.uint8)
            self.labels = test['labels'].astype(np.uint8)

        label_idx_under10 = []
        for idx, label in enumerate(self.labels):
            if int(self.labels[idx]) < 10:
                label_idx_under10.append(idx)

        self_data_tmp = self.data[label_idx_under10]
        self_labels_tmp = self.labels[label_idx_under10]
        self.data = self_data_tmp
        self.labels = self_labels_tmp.astype(np.int64)
        self.targets = self.labels

        #self.targets = self.labels

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()


########################################################################################################################

class notMNIST(torch.utils.data.Dataset):
    """The notMNIST dataset is a image recognition dataset of font glypyhs for the letters A through J useful with simple neural networks. It is quite similar to the classic MNIST dataset of handwritten digits 0 through 9.
    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self, root, train=True,transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = "notmnist.zip"
        self.url = "https://github.com/nkundiushuti/notmnist_convert/blob/master/notmnist.zip?raw=true"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                self.download()

        training_file = 'notmnist_train.pkl'
        testing_file = 'notmnist_test.pkl'
        if train:
            with open(os.path.join(root,training_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # train  = u.load()
                train = pickle.load(f)
            self.data = train['features'].astype(np.uint8)
            self.labels = train['labels'].astype(np.int64)
        else:
            with open(os.path.join(root,testing_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # test  = u.load()
                test = pickle.load(f)

            self.data = test['features'].astype(np.uint8)
            self.labels = test['labels'].astype(np.int64)

        self.targets = self.labels

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img[0])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = torch.cat([img, img, img], axis=0)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()


########################################################################################################################
