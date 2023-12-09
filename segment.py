import datetime
import logging
import os
import sys
import pickle
import shutil
from glob import glob
from abc import ABC, abstractmethod

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib

import monai
import monai.transforms as mt
from monai.data import PersistentDataset, Dataset, DataLoader, decollate_batch
from monai.apps import CrossValidation, get_logger
from monai.utils import set_determinism
from monai.networks.nets import UNet
from monai.engines import EnsembleEvaluator, SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    CheckpointLoader,
    EarlyStopHandler,
    LrScheduleHandler,
    MeanDice,
    ROCAUC,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.losses import DiceLoss
from monai.inferers import SimpleInferer, SlidingWindowInferer

import torch



# constants
SCRATCH_DIR = '/scratch/users/yanghyun/'
PATH_TRAIN = '/home/groups/booil/brainmetshare-3/train'
IMG_TYPES = {
  'bravo': 0,
  'flair': 1,
  't1_gd': 2,
  't1_pre': 3
}
KEYS = ('image', 'label')



# class definitions

# use PersistentDataset so deterministic preprocessing operations aren't repeated
# slow initialization, fast retrieval
# cache stored in disk not memory
class MRIDataset(ABC, PersistentDataset):
    """
    Base class to generate cross validation datasets.
    """

    def __init__(
        self,
        data,
        transform,
        **kwargs
    ) -> None:
        data = self._split_datalist(datalist=data)
        super().__init__(data, transform, **kwargs)

    @abstractmethod
    def _split_datalist(self, datalist):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
        
class Spec:
    def __init__(self, constr, **kwargs):
        self.constr = constr
        self.kwargs = kwargs
        
    def __call__(self):
        return self.constr(**self.kwargs)
    
class ModSpec(Spec):
    def __init__(self, constr, **kwargs):
        super().__init__(constr, **kwargs)
        
class LosSpec(Spec):
    def __init__(self, constr, **kwargs):
        super().__init__(constr, **kwargs)
        
class OptSpec(Spec):
    def __init__(self, constr, **kwargs):
        super().__init__(constr, **kwargs)
        
    def __call__(self, params):
        return self.constr(params, **self.kwargs)
    
class LrsSpec(Spec):
    def __init__(self, constr, **kwargs):
        super().__init__(constr, **kwargs)
    
    def __call__(self, opt):
        return self.constr(opt, **self.kwargs)
        
class SpecComb:
    def __init__(self, spec_name, modspec, losspec, optspec, lrsspec):
        self.spec_name = spec_name
        self.modspec = modspec
        self.losspec = losspec
        self.optspec = optspec
        self.lrsspec = lrsspec
        
    def __call__(self, device):
        model = self.modspec().to(device)
        loss_function = self.losspec()
        optimizer = self.optspec(model.parameters())
        lrscheduler = self.lrsspec(optimizer)
        return model, loss_function, optimizer, lrscheduler
    
    def __repr__(self):
        return self.spec_name
    
    def __str__(self):
        return self.spec_name



# functions

def get_data_dicts(path):
    data = [
    {
      'image': [os.path.join(subj, img_type + '.nii.gz') for img_type in IMG_TYPES],
      'label': os.path.join(subj, 'seg.nii.gz')
    }
    for subj in glob(os.path.join(path, 'Mets_*'))
    ]
    return data


def fix_meta(metatensor):
    """
    fix meta information of metatensor after stacking
    """
    # fix img meta
    a = [metatensor.ndim, *metatensor.shape[1:], metatensor.shape[0]]
    for i, val in enumerate(a):
        metatensor.meta['dim'][i] = val
        metatensor.meta['original_channel_dim'] = -1
    return metatensor


def print_data(metatensor):
    print(metatensor.shape)
    print(metatensor.meta)
    return metatensor

    
class MySlidingWindowInferer(SlidingWindowInferer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, inputs, network, *args, **kwargs):
        print(f"sw input shape: {inputs.shape}")
        print(f"sw output shape: {network(inputs).shape}")
        return super().__call__(inputs, network, *args, **kwargs)
        

class MyUNet(UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        print(f"before forward shape: {x.shape}")
        x = super().forward(x)
        print(f"after forward shape: {x.shape}")
        return x


# main

def main():
    # cpus
    print(f"CPU count: {os.cpu_count()}")
    
    # gpus
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print(f"GPUs: {available_gpus}")
    
    # set seed
    set_determinism(seed=0)
    
    # set logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    get_logger("train_log")
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # paths
    cache_dir = os.path.join(SCRATCH_DIR, 'cache_dir')
    output_path = os.path.join(SCRATCH_DIR, 'outputs')
    log_path = os.path.join(SCRATCH_DIR, 'logs')
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    sys.stdout.flush()
    
    
    # set variables
    #---------------------------------------#
    # get data
    data_dicts = get_data_dicts(PATH_TRAIN)
    ###data_dicts = data_dicts[:5]

    # ratio of classes
    ratio = torch.Tensor([2500]).to(device)
    ###ratio = torch.Tensor([10]).to(device)

    # number of folds
    num = 5
    folds = list(range(num))

    # for data loaders
    num_workers=4
    ###num_workers=0
    batch_size_train=8
    batch_size_val=1

    # specify model
    spec_comb = SpecComb(
        'spec_comb0',
        ModSpec(
            UNet,
            spatial_dims=3, # 3D
            in_channels=4, # 4 modalities
            out_channels=1, # channel for output
            ###channels=(8, 16, 32, 64, 128), # layers
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=2,
            act='PRELU',
            norm=monai.networks.layers.Norm.BATCH,
            dropout=0.1,
            ###bias=True,
            adn_ordering='NDA'
        ),
        LosSpec(
            torch.nn.BCEWithLogitsLoss,
            weight=ratio
            ###DiceLoss,
            ###sigmoid=True
        ),
        OptSpec(
            torch.optim.Adam,
            lr=1e-3
        ),
        LrsSpec(
            torch.optim.lr_scheduler.MultiStepLR,
            milestones=[3, 9], # decrease every step_size epoch by gamma
            gamma=0.1
        )
    )

    # training parameters
    max_epochs = 12
    val_interval = 2 # validate every val_interval epochs
    save_interval = 2 # save checkpoint every save_interval epochs
    
    # pad so that images are divisible by k
    # k == 2^len(layers)
    k = 2**len(spec_comb.modspec.kwargs['channels'])
    
    # sliding window inference parameters
    roi_factor = 0.5
    overlap = 0.5
    sw_batch_size = 4
    
    # if need to resume training
    ###run_id = ""
    checkpoint_paths = []
    checkpoint_paths += [None] * (num - len(checkpoint_paths))
    print(f"checkpoint paths: {checkpoint_paths}")
    assert len(checkpoint_paths) == num
    #---------------------------------------#


    # transformations

    # training
    xform_train = mt.Compose([
        # load images
        mt.LoadImageD(KEYS),
        # make channel the first dimension / add channel dimension if necessary
        mt.EnsureChannelFirstD(KEYS),
        # fix meta
        mt.LambdaD(KEYS, fix_meta),
        # make sure tensor type
        mt.EnsureTypeD(keys=KEYS),
        
        # scale intensity to [0,1]
        mt.ScaleIntensityd(keys="image", channel_wise=True),
        # removes all zero borders to focus on the valid body area of the images and labels
        mt.CropForegroundd(keys=KEYS, source_key="image"),
        # make sure all have same orientation (axcode)
        mt.Orientationd(keys=KEYS, axcodes="RAS"),
        mt.Spacingd(
          keys=KEYS,
          pixdim=(1.0, 1.0, 1.0),
          mode=("bilinear", "nearest"),
        ),
        
        # randomly crop patch samples from big image based on pos / neg ratio
        mt.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        
        # invariant to affine tranformations
        mt.RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=1, #spatial_size=(96, 96, 96),
            rotate_range=(0, 0, np.pi/15),
            scale_range=(0.1, 0.1, 0.1)
        ),
        
        
        mt.Rand3DElasticd(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=1.0,
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            #spatial_size=(96, 96, 96),
            translate_range=(50, 50, 2),
            rotate_range=(np.pi / 36, np.pi / 36, np.pi),
            scale_range=(0.15, 0.15, 0.15),
            padding_mode="border",
        ),
        
        # pad data to be divisible
        mt.DivisiblePadD(keys=KEYS, k=k),
    ])
    # validation
    xform_val = mt.Compose([
        # load images
        mt.LoadImageD(KEYS),
        # make channel the first dimension / add channel dimension if necessary
        mt.EnsureChannelFirstD(KEYS),
        # fix meta
        mt.LambdaD(KEYS, fix_meta),
        # make sure tensor type
        mt.EnsureTypeD(keys=KEYS),
        
        # scale intensity to [0,1]
        mt.ScaleIntensityd(keys="image", channel_wise=True),
        # removes all zero borders to focus on the valid body area of the images and labels
        mt.CropForegroundd(keys=KEYS, source_key="image"),
        # make sure all have same orientation (axcode)
        mt.Orientationd(keys=KEYS, axcodes="RAS"),
        mt.Spacingd(
          keys=KEYS,
          pixdim=(1.0, 1.0, 1.0),
          mode=("bilinear", "nearest"),
        ),
        # pad data to be divisible
        mt.DivisiblePadD(keys=KEYS, k=k),
    ])
    

    # cross validation dataset
    cvdataset = CrossValidation(
        dataset_cls=MRIDataset,
        data=data_dicts,
        nfolds=num,
        transform=xform_train,
        cache_dir=cache_dir
    )

    # clear cache
    for c in glob(os.path.join(cache_dir, '*')):
        os.remove(c)


    # get datasets
    train_dss = [cvdataset.get_dataset(folds=folds[0:i] + folds[(i + 1) :]) for i in folds]
    val_dss = [cvdataset.get_dataset(folds=i, transform=xform_val) for i in folds]

    # get loaders & set batch size, number of workers, shuffle
    train_loaders = [DataLoader(train_dss[i], batch_size=batch_size_train, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()) for i in folds]
    val_loaders = [DataLoader(val_dss[i], batch_size=batch_size_val, num_workers=num_workers, pin_memory=torch.cuda.is_available()) for i in folds]


    # get shape of image
    sample = next(iter(train_loaders[0]))['image']
    print(f"input shape: {sample.shape}")
    
    # set roi size
    offset = 2
    roi_size = tuple(int(sample.shape[i] * roi_factor) for i in range(offset, len(sample.shape)))
    # (input/roi) is divisible by one or the other
    print(f"roi_size: {roi_size}")
    for i in range(offset, len(sample.shape)):
        if not (sample.shape[i] % roi_size[i - offset] == 0 or roi_size[i - offset] % sample.shape[i] == 0):
            raise ValueError(f"{sample.shape[i]}, {roi_size[i - offset]} not divisible")


    # train function
    def train(spec_comb, index=0, checkpoint_path=None):
        # instantiate net, loss, opt
        net, loss, opt, lr_scheduler = spec_comb(device)
        
        print(f"Model on CUDA? {next(net.parameters()).device}")
        print(net)

        # post processing transformations for validation
        val_post_transforms = mt.Compose([
            mt.EnsureTyped(keys="pred"), # ensure tensor type
            mt.Activationsd(keys="pred", sigmoid=True), # take sigmoid activation of values
            mt.AsDiscreted(keys="pred", threshold=0.5)] # turn into 0, 1
        )

        
        # configure additional things to do during validation
        val_handlers = [
            # apply “EarlyStop” logic based on the validation metrics
            ###EarlyStopHandler(trainer=None, patience=10, score_function=lambda x: x.state.metrics["val_mean_dice"]),
            # use the logger "train_log" defined at the beginning of this program
            # for simple logging
            StatsHandler(name="train_log", output_transform=lambda x: None),
            # write tensorboard logs
            TensorBoardStatsHandler(log_dir=os.path.join(log_path, run_id, str(index)), output_transform=lambda x: None),
            # write tensorboard log images
            TensorBoardImageHandler(
                log_dir=os.path.join(log_path, run_id, str(index)),
                batch_transform=from_engine(["image", "label"]),
                output_transform=from_engine(["pred"]),
            ),
        ]

        # configure validation phase
        evaluator = SupervisedEvaluator(
            device=device, # device
            val_data_loader=val_loaders[index], # validation data
            network=net, # network
            inferer=SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap), # infer using sw
            postprocessing=val_post_transforms, # post processing
            # metric
            key_val_metric={
                "val_mean_dice": MeanDice(
                    output_transform=from_engine(["pred", "label"]),
                    num_classes=2
                )
            },
            ###additional_metrics={
            ###    "val_rocauc": ROCAUC(output_transform=from_engine(["pred", "label"]))
            ###},
            val_handlers=val_handlers, # additional things to do during validation
            amp=True, # enable auto mixed precision for performance boost
        )
        
        # post processing transformations for training
        ###train_post_transforms = mt.Compose([
        ###    mt.Activationsd(keys="pred", sigmoid=True), # take sigmoid activation of values
        ###    mt.AsDiscreted(keys="pred", threshold=0.5) # turn into 0, 1
        ###])
        
        # additional things to do during training
        train_handlers = [
            # apply “EarlyStop” logic based on the loss value, use “-” negative value because smaller loss is better
            # early stop based on change in loss over epoch
            EarlyStopHandler(
                trainer=None, patience=10, score_function=lambda x: -x.state.output[0]["loss"], epoch_level=True
            ),
            LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True), # handle learning rate
            ValidationHandler(validator=evaluator, interval=val_interval, epoch_level=True), # validate
            # use the logger "train_log" defined at the beginning of this program
            StatsHandler(name="train_log", tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
            # tensorboard log
            TensorBoardStatsHandler(
                log_dir=os.path.join(log_path, run_id, str(index)), tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
            ),
        ]

        # configure training phase
        trainer = SupervisedTrainer(
            device=device, # device
            max_epochs=max_epochs, # epochs
            train_data_loader=train_loaders[index], # train data
            network=net, # network
            optimizer=opt, # optimizer
            loss_function=loss, # loss
            inferer=SimpleInferer(), # infer using forward() directly
            amp=True, # amp
            train_handlers=train_handlers, # additional things to do during training
            # in case wanting to use following metric to save checkpoint
            ###postprocessing=train_post_transforms,
            ###key_train_metric={"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
        )
        
        # things to save
        save_dict={
            "net": net,
            "opt": opt,
            "lf": loss,
            "lrs": lr_scheduler,
            "trainer": trainer
        }
        
        # add save handler for training
        saver_train = CheckpointSaver(
            save_dir=os.path.join(output_path, run_id, str(index)),
            # save net, opt, loss
            save_dict=save_dict,
            file_prefix=spec_comb.spec_name + "_cv=" + str(index), # prefix
            save_final=True, # save at end
            save_interval=save_interval, epoch_level=True # save every save_interval epochs
        )
        saver_train.attach(trainer)
        # add save handler for validation
        saver_val = CheckpointSaver(
            save_dir=os.path.join(output_path, run_id, str(index)),
            # save net, opt, loss
            save_dict=save_dict,
            file_prefix=spec_comb.spec_name + "_cv=" + str(index), # prefix
            key_metric_save_state=True, # save the tracking list of key metric
            save_key_metric=True, # save when best metric
        )
        saver_val.attach(evaluator)
        
        
        # set initialized trainer for "early stop" handlers
        ###val_handlers[0].set_trainer(trainer=trainer)
        train_handlers[0].set_trainer(trainer=trainer)
        # if loading
        if checkpoint_path:
            handler = CheckpointLoader(load_path=checkpoint_path, load_dict=save_dict, map_location=device)
            handler(trainer)
        # run training
        trainer.run()
        
        return net
        

    # train model
    for idx, path in enumerate(checkpoint_paths):
        train(spec_comb, index=idx, checkpoint_path=path)
    
    # remove cache
    shutil.rmtree(cache_dir)


if __name__ == '__main__':
    main()
    
