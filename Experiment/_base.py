import random

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import IS2D_model
from utils.get_functions import *
from dataset.PolypSeg import PolypImageSegDataset

class BaseSegmentationExperiment(object):
    def __init__(self, args):
        super(BaseSegmentationExperiment, self).__init__()

        self.args = args
        self.device = get_deivce()
        self.fix_seed(self.device)
        self.history_generator()
        self.scaler = torch.cuda.amp.GradScaler()
        self.start, self.end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.inference_time_list = []
        self.metric_list = ['Loss', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'IoU']

        print("STEP1. Load {} Dataset Loader...".format(args.data_type))
        if args.data_type in ['CVC-ClinicDB', 'Kvasir-SEG', 'BKAI-IGH-NeoPolyp'] :
            train_dataset = PolypImageSegDataset(args.dataset_dir, mode='train', transform=self.transform_generator('train')[0], target_transform=self.transform_generator('train')[1])
            val_dataset = PolypImageSegDataset(args.dataset_dir, mode='val', transform=self.transform_generator('test')[0], target_transform=self.transform_generator('test')[1])
            test_dataset = PolypImageSegDataset(args.dataset_dir, mode='test', transform=self.transform_generator('test')[0], target_transform=self.transform_generator('test')[1])

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=int(args.batch_size / args.ngpus_per_node) if args.distributed else args.batch_size,
                                       shuffle=(self.train_sampler is None),
                                       num_workers=int((args.num_workers+args.ngpus_per_node-1)/args.ngpus_per_node) if args.distributed else args.num_workers,
                                       pin_memory=True, sampler=self.train_sampler)
        self.val_loader = DataLoader(val_dataset,
                                      batch_size=int(args.batch_size / args.ngpus_per_node) if args.distributed else args.batch_size,
                                      shuffle=False,
                                      num_workers=int((args.num_workers+args.ngpus_per_node-1)/args.ngpus_per_node) if args.distributed else args.num_workers,
                                      pin_memory=True)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=int(args.batch_size / args.ngpus_per_node) if args.distributed else args.batch_size,
                                      shuffle=False,
                                      num_workers=int((args.num_workers+args.ngpus_per_node-1)/args.ngpus_per_node) if args.distributed else args.num_workers,
                                      pin_memory=True)

        print("STEP2. Load 2D Image Segmentation Model {}...".format(args.model_name))
        self.model = IS2D_model(self.device, args.model_name, args.image_size, args.num_channels)
        if args.distributed:
            print('Multi GPU activate : {} with DP'.format(torch.cuda.device_count()))
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                self.model.cuda(args.gpu)
                # when using a single GPU per process and per DDP, we need to divide tha batch size
                # ourselves based on the total number of GPUs we have
                self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
            else:
                self.model.cuda()
                # DDP will divide and allocate batch_size to all available GPUs if device_ids are not set
                # 만약에 device_ids를 따로 설정해주지 않으면, 가능한 모든 gpu를 기준으로 ddp가 알아서 배치사이즈와 workers를 나눠준다는 뜻.
                self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        elif not args.distributed and torch.cuda.device_count() > 1:
            print('Multi GPU activate : {}'.format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)

        print("STEP3. Load Optimizer {}...".format(args.optimizer_name))
        self.optimizer = get_optimizer(args.optimizer_name, self.model, args.lr, args.momentum, args.weight_decay)

        print("STEP4. Load LRS {}...".format(args.LRS_name))
        self.scheduler = get_scheduler(args.LRS_name, self.optimizer, args.final_epoch, len(self.train_loader), args.lr)

        print("STEP5. Load Criterion {}...".format(args.criterion))
        self.criterion = get_criterion(args.criterion)

        if args.distributed: self.criterion.cuda(args.gpu)

    def print_params(self):
        print("\ndata type : {}".format(self.args.data_type))
        print("model : {}".format(self.args.model_name))
        print("optimizer : {}".format(self.optimizer))
        print("learning rate : {}".format(self.args.lr))
        print("learning rate scheduler : {}".format(self.args.LRS_name))
        print("start epoch : {}".format(self.args.start_epoch))
        print("final epoch : {}".format(self.args.final_epoch))
        print("criterion : {}".format(self.criterion))
        print("batch size : {}".format(self.args.batch_size))
        print("image size : ({}, {}, {})".format(self.args.image_size, self.args.image_size, self.args.num_channels))

    def fix_seed(self, device):
        random.seed(4321)
        np.random.seed(4321)
        torch.manual_seed(4321)
        torch.cuda.manual_seed(4321)
        torch.cuda.manual_seed_all(4321)
        if self.args.reproducibility :
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print("your seed is fixed to '4321' with reproducibility {}".format(self.args.reproducibility))

    def history_generator(self):
        self.history = dict()
        self.history['train_loss'] = list()
        self.history['val_loss'] = list()

    def transform_generator(self, mode):
        if mode == 'train' :
            transform_list = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation((-5, 5), expand=False),
                transforms.ToTensor(),
            ]

            target_transform_list = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation((-5, 5), expand=False),
                transforms.ToTensor(),
            ]

        else :
            transform_list = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
            ]

            target_transform_list = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),

            ]

        return transforms.Compose(transform_list), transforms.Compose(target_transform_list)

    def current_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def forward(self, image, target):
        if self.args.distributed: image, target = image.cuda(), target.cuda()
        else: image, target = image.to(self.device), target.to(self.device)

        with torch.cuda.amp.autocast():
            output = self.model(image)
            loss = self.model._calculate_criterion(output, target)

        return loss, output, target

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.args.LRS_name == 'CALRS': self.scheduler.step()