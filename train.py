"""
Training loop for NICEModel. Attempts to replicate the conditions in the NICE paper.

Supports the following datasets:
* MNIST (LeCun & Cortes, 1998);
* Toronto Face Dataset (Susskind et al, 2010);
* CIFAR-10 (Krizhevsky, 2010);
* Street View House Numbers (Netzer et al, 2011).

We apply a dequantization for MNIST, TFD, SVHN as follows (following the NICE authors):
1. Add uniform noise ~ Unif([0, 1/256]);
2. Rescale data to be in [0,1] in each dimension.

For CIFAR10, we instead do:
1. Add uniform noise ~ Unif([-1/256, 1/256]);
2. Rescale data to be in [-1,1] in each dimensions.

Additionally, we perform:
* approximate whitening for TFD;
* exact ZCA on SVHN, CIFAR10;
* no additional preprocessing for MNIST.

Finally, images are flattened from (H,W) to (H*W,).
"""
# numeric/nn libraries:
import torch
import torchvision
import torch.optim as optim
import torch
import torch.utils.data as data
import numpy as np
# models/losses/image utils:
from nice.models import NICEModel
from nice.loss import LogisticPriorNICELoss, GaussianPriorNICELoss, BinomialPriorNICELoss
from nice.utils import rescale, l1_norm
# python/os utils:
import argparse
import os
from tqdm import tqdm, trange
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import shutil
from datetime import date
from nice.utils import get_random_string
from sklearn.datasets import make_moons
from torch.utils.tensorboard import SummaryWriter

# set CUDA training on if detected:
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    CUDA = True
else:
    DEVICE = torch.device('cpu')
    CUDA = False

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Dataset loaders: each of these helper functions does the following:
# 1) downloads the corresponding dataset into a folder (if not already downloaded);
# 2) adds the corresponding whitening & rescaling transforms;
# 3) returns a dataloader for that dataset.


def load_moons(train=True, batch_size=1, num_workers=0, alpha=0.05):
    rng = np.random.RandomState(42)
    n_inliers_train = int(10000 * (1 - alpha))
    n_outliers_train = int(10000 * alpha)
    n_inliers_test = int(0.1 * n_inliers_train)
    n_outliers_test = int(0.1 * n_outliers_train)

    X_ok = 4. * (make_moons(n_samples=n_inliers_train, noise=.05, random_state=0)[0] - np.array([0.5, 0.25]))
    X_out = rng.uniform(low=-6, high=6, size=(n_outliers_train, 2))
    X = np.concatenate((X_ok, X_out))
    y_ok = np.array([0] * n_inliers_train)
    y_out = np.array([-1] * n_outliers_train)
    y = np.concatenate((y_ok, y_out))
    X, y = shuffle(X, y)

    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    train_dataset = data.TensorDataset(tensor_X, tensor_y)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=CUDA, drop_last=False)

    X_ok_test = 4. * (make_moons(n_samples=n_inliers_test, noise=.05, random_state=0)[0] - np.array([0.5, 0.25]))
    X_out_test = rng.uniform(low=-6, high=6, size=(n_outliers_test, 2))
    X_test = np.concatenate((X_ok_test, X_out_test))
    y_ok_test = np.array([0] * n_inliers_test)
    y_out_test = np.array([-1] * n_outliers_test)
    y_test = np.concatenate((y_ok_test, y_out_test))

    tensor_X_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)
    test_dataset = data.TensorDataset(tensor_X_test, tensor_y_test)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=CUDA, drop_last=False)

    if train:
        return train_dataloader
    else:
        return test_dataloader


def load_gausses(train=True, batch_size=1, num_workers=0, alpha=0.05):
    n_train = int(10000)
    n_test = int(0.1*10000)

    X_gauss1 = np.random.multivariate_normal(mean=[-5.0, 0.0], cov=np.eye(2), size=int(n_train/2))
    X_gauss2 = np.random.multivariate_normal(mean=[5.0, 0.0], cov=np.eye(2), size=int(n_train/2))
    X = np.concatenate((X_gauss1, X_gauss2))
    y_gauss1 = np.array([0] * int(n_train/2))
    y_gauss2 = np.array([-1] * int(n_train / 2))
    y = np.concatenate((y_gauss1, y_gauss2))
    X, y = shuffle(X, y)

    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    train_dataset = data.TensorDataset(tensor_X, tensor_y)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=CUDA, drop_last=False)

    X_gauss1_test = np.random.multivariate_normal(mean=[-3.0, 0.0], cov=np.eye(2), size=int(n_test / 2))
    X_gauss2_test = np.random.multivariate_normal(mean=[3.0, 0.0], cov=np.eye(2), size=int(n_test / 2))
    X_test = np.concatenate((X_gauss1_test, X_gauss2_test))
    y_gauss1_test = np.array([0] * int(n_test / 2))
    y_gauss2_test = np.array([-1] * int(n_test / 2))
    y_test = np.concatenate((y_gauss1_test, y_gauss2_test))
    X_test, y_test = shuffle(X_test, y_test)

    tensor_X_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)
    test_dataset = data.TensorDataset(tensor_X_test, tensor_y_test)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=CUDA, drop_last=False)

    if train:
        return train_dataloader
    else:
        return test_dataloader


def load_blobs(train=True, batch_size=1, num_workers=0, alpha=0.05):
    n_inliers_train = int(10000*(1-alpha))
    n_outliers_train = int(10000*alpha)
    n_inliers_test = int(0.1*n_inliers_train)
    n_outliers_test = int(0.1*n_outliers_train)
    X_ok = np.random.multivariate_normal(mean=[0.0, 0.0], cov=np.eye(2), size=n_inliers_train)
    X_outlier = np.random.multivariate_normal(mean=[4.0, 4.0], cov=0.3 * np.eye(2), size=n_outliers_train)
    X = np.concatenate((X_ok, X_outlier))
    y_ok = np.array([0] * n_inliers_train)
    y_outlier = np.array([-1] * n_outliers_train)

    y = np.concatenate((y_ok, y_outlier))
    X, y = shuffle(X, y)

    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    train_dataset = data.TensorDataset(tensor_X, tensor_y)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=CUDA, drop_last=False)

    X_ok_test = np.random.multivariate_normal(mean=[0.0, 0.0], cov=np.eye(2), size=n_inliers_test)
    X_outlier_test = np.random.multivariate_normal(mean=[4.0, 4.0], cov=0.3 * np.eye(2), size=n_outliers_test)
    X_test = np.concatenate((X_ok_test, X_outlier_test))
    y_ok_test = np.array([0] * n_inliers_test)
    y_outlier_test = np.array([-1] * n_outliers_test)

    y_test = np.concatenate((y_ok_test, y_outlier_test))

    tensor_X_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)
    test_dataset = data.TensorDataset(tensor_X_test, tensor_y_test)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=CUDA, drop_last=False)

    if train:
        return train_dataloader
    else:
        return test_dataloader


def load_mnist(train=True, batch_size=1, num_workers=0, alpha=0.05):
    """Rescale and preprocess MNIST dataset."""
    mnist_transform = torchvision.transforms.Compose([
        # convert PIL image to tensor:
        torchvision.transforms.ToTensor(),
        # flatten:
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
        # add uniform noise:
        torchvision.transforms.Lambda(lambda x: (x + torch.rand_like(x).div_(256.))),
        # rescale to [0,1]:
        torchvision.transforms.Lambda(lambda x: rescale(x, 0., 1.))
    ])
    print(torchvision.datasets.MNIST(root="./datasets/mnist", train=train, transform=mnist_transform, download=False))
    return data.DataLoader(
        torchvision.datasets.MNIST(root="./datasets/mnist", train=train, transform=mnist_transform, download=False),
        batch_size=batch_size,
        pin_memory=CUDA,
        drop_last=train
    )

def load_svhn(train=True, batch_size=1, num_workers=0, alpha=0.05):
    """Rescale and preprocess SVHN dataset."""
    # check if ZCA matrix exists on dataset yet:
    assert os.path.exists("./datasets/svhn/zca_matrix.pt"), \
        "[load_svhn] ZCA whitening matrix not built! Run `python make_dataset.py` first."
    zca_matrix = torch.load("./datasets/svhn/zca_matrix.pt")

    svhn_transform = torchvision.transforms.Compose([
        # convert PIL image to tensor:
        torchvision.transforms.ToTensor(),
        # flatten:
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
        # add uniform noise:
        torchvision.transforms.Lambda(lambda x: (x + torch.rand_like(x).div_(256.))),
        # rescale to [0,1]:
        torchvision.transforms.Lambda(lambda x: rescale(x, 0., 1.)),
        # exact ZCA:
        torchvision.transforms.LinearTransformation(zca_matrix)
    ])
    _mode = 'train' if train else 'test'
    return data.DataLoader(
        torchvision.datasets.SVHN(root="./datasets/svhn", split=_mode, transform=svhn_transform, download=False),
        batch_size=batch_size,
        pin_memory=CUDA,
        drop_last=train
    )

def load_cifar10(train=True, batch_size=1, num_workers=0, alpha=0.05):
    """Rescale and preprocess CIFAR10 dataset."""
    # check if ZCA matrix exists on dataset yet:
    assert os.path.exists("./datasets/cifar/zca_matrix.pt"), \
        "[load_cifar10] ZCA whitening matrix not built! Run `python make_datasets.py` first."
    zca_matrix = torch.load("./datasets/cifar/zca_matrix.pt")

    cifar10_transform = torchvision.transforms.Compose([
        # convert PIL image to tensor:
        torchvision.transforms.ToTensor(),
        # flatten:
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
        # add uniform noise ~ [-1/256, +1/256]:
        torchvision.transforms.Lambda(lambda x: (x + torch.rand_like(x).div_(128.).add_(-1./256.))),
        # rescale to [-1,1]:
        torchvision.transforms.Lambda(lambda x: rescale(x,-1.,1.)),
        # exact ZCA:
        torchvision.transforms.LinearTransformation(zca_matrix)
    ])
    return data.DataLoader(
        torchvision.datasets.CIFAR10(root="./datasets/cifar", train=train, transform=cifar10_transform, download=False),
        batch_size=batch_size,
        pin_memory=CUDA,
        drop_last=train
    )

def load_tfd(train=True, batch_size=1, num_workers=0, alpha=0.05):
    """Rescale and preprocess TFD dataset."""
    raise NotImplementedError("[load_tfd] Toronto Faces Dataset unsupported right now. Sorry!")

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Training loop: return a NICE model trained over a number of epochs.
def train(args):
    """Construct a NICE model and train over a number of epochs."""
    # === choose which dataset to build:
    if args.dataset == 'blobs':
        dataloader_fn = load_blobs
        input_dim = 2
    if args.dataset == 'moons':
        dataloader_fn = load_moons
        input_dim = 2
    if args.dataset == 'gausses':
        dataloader_fn = load_gausses
        input_dim = 2
    if args.dataset == 'mnist':
        dataloader_fn = load_mnist
        input_dim = 28*28
    if args.dataset == 'svhn':
        dataloader_fn = load_svhn
        input_dim = 32*32*3
    if args.dataset == 'cifar10':
        dataloader_fn = load_cifar10
        input_dim = 32*32*3
    if args.dataset == 'tfd':
        raise NotImplementedError("[train] Toronto Faces Dataset unsupported right now. Sorry!")
        dataloader_fn = load_tfd
        input_dim = None

    # === build model & optimizer:
    model = NICEModel(input_dim, args.nhidden, args.nlayers)
    if (args.model_path is not None):
        assert(os.path.exists(args.model_path)), "[train] model does not exist at specified location"
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1,args.beta2), eps=args.eps)

    # === choose which loss function to build:
    if args.prior == 'logistic':
        nice_loss_fn = LogisticPriorNICELoss(size_average=True)
    elif args.prior == 'binomial':
        nice_loss_fn = BinomialPriorNICELoss(size_average=True)
    else:
        nice_loss_fn = GaussianPriorNICELoss(size_average=True)

    def loss_fn(fx, DEVICE):
        """Compute NICE loss w/r/t a prior and optional L1 regularization."""
        if args.lmbda == 0.0:
            return nice_loss_fn(fx, model.scaling_diag, DEVICE, args.alpha)
        else:
            return nice_loss_fn(fx, model.scaling_diag, DEVICE, args.alpha) + args.lmbda*l1_norm(model, include_bias=True)

    # === train over a number of epochs; perform validation after each:
    path = date.today().strftime('%m_%d_')+\
           '_dataset={}_alpha={}_prior={}_batch_size={}_nlayers={}_nhidden={}_epochs={}_'.format(args.dataset, args.alpha, args.prior, args.batch_size, args.nlayers, args.nhidden, args.num_epochs)+get_random_string()
    path_plots = 'runs/'+str(args.dataset)+'/'+path
    path_tensorboard = 'logs/'+str(args.dataset)+'/'+path
    if os.path.isdir(path_plots):
        shutil.rmtree(path_plots)
    os.makedirs(path_plots)

    writer = SummaryWriter(log_dir=path_tensorboard)

    for t in range(args.num_epochs):
        print("* Epoch {0}:".format(t))
        dataloader = dataloader_fn(train=True, batch_size=args.batch_size, alpha=args.alpha)
        losses = []
        last_loss = 0.0
        for inputs, _ in tqdm(dataloader):
            opt.zero_grad()
            loss = loss_fn(model(inputs.to(DEVICE)), DEVICE)
            a = loss
            a = a.cpu().detach().numpy()
            loss.backward()
            opt.step()
            last_loss = a
            losses.append(a)
        writer.add_scalar('Loss/train_mean', np.mean(np.array(losses)), t+1)
        writer.add_scalar('Loss/train', last_loss, t+1)
        
        # save model to disk and delete dataloader to save memory:
        if t % args.save_epoch == 0 and args.save:
            _dev = 'cuda' if CUDA else 'cpu'
            _fn = "nice.{0}.l_{1}.h_{2}.p_{3}.e_{4}.{5}.pt".format(args.dataset, args.nlayers, args.nhidden, args.prior, t, _dev)
            torch.save(model.state_dict(), os.path.join(args.savedir, _fn))
            print(">>> Saved file: {0}".format(_fn))
        del dataloader
        
        # perform validation loop:
        vmin, vmed, vmean, vmax = validate(model, dataloader_fn, nice_loss_fn, args.alpha)
        print(">>> Validation Loss Statistics: min={0}, med={1}, mean={2}, max={3}".format(vmin,vmed,vmean,vmax))
        if args.dataset in ['blobs', 'moons', 'gausses']:
            validate_outliers(model, dataloader_fn, t+1, path_plots, args.alpha)
        writer.add_scalar('Validation/vmin', vmin, t+1)
        writer.add_scalar('Validation/vmed', vmed, t+1)
        writer.add_scalar('Validation/vmean', vmean, t+1)
        writer.add_scalar('Validation/vmax', vmax, t+1)
    writer.close()


def validate_outliers(model, dataloader_fn, epoch, path, alpha):
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    model.eval()
    dataloader = dataloader_fn(train=False, batch_size=args.batch_size, alpha=alpha)

    validation = []
    targets = []
    inverses = []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            validation.append(model.forward(x.to(DEVICE)).cpu().detach().numpy())
            inverses.append(model.inverse(model.forward(x.to(DEVICE))).cpu().detach().numpy())
            targets.append(y)
        validation = np.concatenate(validation, axis=0)
        inverses = np.concatenate(inverses, axis=0)
        targets = torch.cat(targets).cpu().detach().numpy()
        latent_x, latent_y = np.hsplit(validation, 2)
        inverse_x, inverse_y = np.hsplit(inverses, 2)
        norms = np.sqrt(np.power(latent_x, 2) + np.power(latent_y, 2))
        #norm95 = np.percentile(norms, 95)
        #norm98 = np.percentile(norms, 98)
        treshold = np.percentile(norms, int((1-alpha)*100))
        colors = []
        for n in norms:
            if n <= treshold:
                colors.append('black')
            else:
                colors.append('red')

        ax1.title.set_text('Inverse')
        ax2.title.set_text('Flow')
        ax1.scatter(inverse_x, inverse_y, c=colors)
        ax2.scatter(latent_x, latent_y, c=colors)

        fig.savefig('./{}/epoch_{}.png'.format(path, epoch))

    del dataloader

    return
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Validation loop: set gradient-tracking off with model in eval mode:
def validate(model, dataloader_fn, loss_fn, alpha):
    """Perform validation on a dataset."""
    # set model to eval mode (turns batch norm training off)
    model.eval()

    # build dataloader in eval mode:
    dataloader = dataloader_fn(train=False, batch_size=args.batch_size, alpha=alpha)

    # turn gradient-tracking off (for speed) during validation:
    validation_losses = []
    with torch.no_grad():
        for inputs,_ in tqdm(dataloader):
            validation_losses.append(loss_fn(model(inputs.to(DEVICE)), model.scaling_diag, DEVICE, args.alpha).item())
    
    # delete dataloader to save memory:
    del dataloader

    # set model back in train mode:
    model.train()

    # return validation loss summary statistics:
    return (np.amin(validation_losses),
            np.median(validation_losses),
            np.mean(validation_losses),
            np.amax(validation_losses))

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
if __name__ == '__main__':
    # ----- parse training settings:
    parser = argparse.ArgumentParser(description="Train a fresh NICE model and save.")
    # configuration settings:
    parser.add_argument("--dataset", required=True, dest='dataset', choices=('tfd', 'cifar10', 'svhn', 'mnist', 'blobs', 'moons', 'gausses'),
                        help="Dataset to train the NICE model on.")
    parser.add_argument("--epochs", dest='num_epochs', default=1500, type=int,
                        help="Number of epochs to train on. [1500]")
    parser.add_argument("--batch_size", dest="batch_size", default=16, type=int,
                        help="Number of examples per batch. [16]")
    parser.add_argument("--save_epoch", dest="save_epoch", default=10, type=int,
                        help="Number of epochs between saves. [10]")
    parser.add_argument("--savedir", dest='savedir', default="./saved_models",
                        help="Where to save the trained model. [./saved_models]")
    # model settings:
    parser.add_argument("--nonlinearity_layers", dest='nlayers', default=5, type=int,
                        help="Number of layers in the nonlinearity. [5]")
    parser.add_argument("--nonlinearity_hiddens", dest='nhidden', default=1000, type=int,
                        help="Hidden size of inner layers of nonlinearity. [1000]")
    parser.add_argument("--prior", choices=('binomial', 'logistic', 'gaussian'), default="logistic",
                        help="Prior distribution of latent space components. [logistic]")
    parser.add_argument("--model_path", dest='model_path', default=None, type=str,
                        help="Continue from pretrained model. [None]")
    # optimization settings:
    parser.add_argument("--lr", default=0.001, dest='lr', type=float,
                        help="Learning rate for ADAM optimizer. [0.001]")
    parser.add_argument("--beta1", default=0.9,  dest='beta1', type=float,
                        help="Momentum for ADAM optimizer. [0.9]")
    parser.add_argument("--beta2", default=0.01, dest='beta2', type=float,
                        help="Beta2 for ADAM optimizer. [0.01]")
    parser.add_argument("--eps", default=0.0001, dest='eps', type=float,
                        help="Epsilon for ADAM optimizer. [0.0001]")
    parser.add_argument("--lambda", default=0.0, dest='lmbda', type=float,
                        help="L1 weight decay coefficient. [0.0]")
    parser.add_argument("--save", default=False, type=bool,
                        help="If save models?")
    parser.add_argument("--alpha", default=0.05, dest='alpha', type=float,
                        help="Percent of outliers")
    args = parser.parse_args()
    # ----- run training loop over several epochs & save models for each epoch:
    model = train(args)
