from time import time

import torch

import numpy as np
from tqdm import tqdm

from utils.calculate_metrics import metrics
from ._base import BaseSegmentationExperiment

class Biomedical2DImageSegmentationExperiment(BaseSegmentationExperiment):
    def __init__(self, args):
        super(Biomedical2DImageSegmentationExperiment, self).__init__(args)
        self.count = 1

    def fit(self):
        self.print_params()
        if self.args.train:
            for epoch in tqdm(range(self.args.start_epoch, self.args.final_epoch + 1)):
                print('\n============ EPOCH {}/{} ============\n'.format(epoch, self.args.final_epoch))
                if self.args.distributed: self.train_sampler.set_epoch(epoch)
                if epoch % 10 == 0: self.print_params()

                epoch_start_time = time()

                print("TRAINING")
                train_results = self.train_epoch(epoch)

                print("EVALUATE")
                val_results = self.val_epoch(epoch)

                total_epoch_time = time() - epoch_start_time
                m, s = divmod(total_epoch_time, 60)
                h, m = divmod(m, 60)

                self.history['train_loss'].append(train_results)
                self.history['val_loss'].append(val_results)

                print('\nEpoch {}/{} : train loss {} | val loss {} | current lr {} | took {} h {} m {} s'.format(
                    epoch, self.args.final_epoch, np.round(train_results, 4), np.round(val_results, 4),
                    self.current_lr(self.optimizer), int(h), int(m), int(s)))

            print("INFERENCE")
            test_results = self.inference(self.args.final_epoch)
            return self.model, self.optimizer, self.scheduler, self.history, test_results, self.metric_list
        else :
            print("INFERENCE")
            test_results = self.inference(self.args.final_epoch)

            return test_results

    def train_epoch(self, epoch):
        self.model.train()

        total_loss, total = 0., 0
        for batch_idx, (image, target) in enumerate(self.train_loader):
            loss, output, target = self.forward(image, target)

            self.backward(loss)

            total_loss += loss.item() * image.size(0)
            total += image.size(0)

            if (batch_idx + 1) % self.args.step == 0 or (batch_idx + 1) == len(self.train_loader):
                print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                    epoch, batch_idx + 1, len(self.train_loader),
                    np.round((batch_idx + 1) / len(self.train_loader) * 100.0, 2), total_loss / total))

        train_loss = total_loss / total
        self.count = 0

        return train_loss

    def val_epoch(self, epoch):
        self.model.eval()
        self.spectrum_ratio = 1
        total_loss, total = .0, 0

        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.val_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                loss, output, target = self.forward(image, target)

                total_loss += loss.item() * image.size(0)
                total += target.size(0)

        val_loss = total_loss / total

        return val_loss

    def inference(self, epoch):
        self.model.eval()

        total_loss, total = .0, 0
        accuracy_list, f1_score_list, precision_list, recall_list, iou_list = [], [], [], [], []

        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.test_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                self.start.record()
                loss, output, target = self.forward(image, target)

                self.end.record()
                torch.cuda.synchronize()
                self.inference_time_list.append(self.start.elapsed_time(self.end))

                for target_, output_ in zip(target, output) :
                    predict = torch.sigmoid(output_).squeeze()
                    accuracy, f1_score, precision, recall, iou = metrics(target_, predict)
                    accuracy_list.append(accuracy); f1_score_list.append(f1_score); precision_list.append(precision), recall_list.append(recall), iou_list.append(iou)

                total_loss += loss.item() * image.size(0)
                total += target.size(0)

        test_loss = total_loss / total
        accuracy = np.round(np.mean(accuracy_list), 4)
        f1_score = np.round(np.mean(f1_score_list), 4)
        precision = np.round(np.mean(precision_list), 4)
        recall = np.round(np.mean(recall_list), 4)
        iou = np.round(np.mean(iou_list), 4)

        if self.args.final_epoch == epoch : print("Mean Inference Time (ms) : {} ({})".format(np.mean(self.inference_time_list), np.std(self.inference_time_list)))

        return test_loss, accuracy, f1_score, precision, recall, iou