import os

import torch

import numpy as np
import pandas as pd

from .get_functions import get_save_path
# from .plot_functions import plot_loss_acc

def save_result(args, model, optimizer, scheduler, history, test_results, current_epoch, best_results, metric_list):
    if (args.distributed and torch.distributed.get_rank() == 0) or not args.multiprocessing_distributed:
        model_dirs = get_save_path(args)

        print("Your experiment is saved in {}.".format(model_dirs))

        print("STEP1. Save {} Model Weight...".format(args.model_name))
        save_model(model, optimizer, scheduler, model_dirs, current_epoch)

        print("STEP2. Save {} Model Test Results...".format(args.model_name))
        save_metrics(test_results, model_dirs, current_epoch, metric_list)

        if args.final_epoch == current_epoch:
            print("STEP3. Save {} Model History...".format(args.model_name))
            save_loss(history, model_dirs)

        print("Current EPOCH {} model is successfully saved at {}".format(current_epoch, model_dirs))

def save_model(model, optimizer, scheduler, model_dirs, current_epoch):
    check_point = {
        'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'current_epoch': current_epoch
    }

    torch.save(check_point, os.path.join(model_dirs, 'model_weights/model_weight(EPOCH {}).pth.tar'.format(current_epoch)))

def save_metrics(test_results, model_dirs, current_epoch, metric_list):
    test_results_dict = dict()
    for metric, result in zip(metric_list, test_results):
        test_results_dict[metric] = result
    # test_loss  = test_results

    # best_top1_error, best_top5_error = best_results

    print("###################### TEST REPORT ######################")
    for metric in metric_list:
        print("test {}\t       :\t {}".format(metric, np.round(test_results_dict[metric], 4)))
    print("###################### TEST REPORT ######################")

    test_results_save_path = os.path.join(model_dirs, 'test_reports', 'test_report(EPOCH {}).txt'.format(current_epoch))
    print(test_results_save_path)
    f = open(test_results_save_path, 'w')

    f.write("###################### TEST REPORT ######################\n")
    for metric in metric_list:
        f.write("test {}\t       :\t {}\n".format(metric, np.round(test_results_dict[metric], 4)))
    f.write("###################### TEST REPORT ######################\n")

    f.close()

def save_loss(history, model_dirs):
    pd.DataFrame(history).to_csv(os.path.join(model_dirs, 'loss.csv'), index=False)