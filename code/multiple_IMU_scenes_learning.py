import cv2  # DO NOT REMOVE
from datasets import SceneData, ScenesDataSet
import train
from utils import general_utils, path_utils
from utils.Phases import Phases
import torch
import loss_functions
from time import time
import math


def train_single_model(conf, device, phase):
    # Create model
    model = general_utils.get_class("models." + conf.get_string("model.type"))(conf).to(device)
    if phase is Phases.FINE_TUNE:
        path = path_utils.path_to_model(conf, Phases.TRAINING)
        model.load_state_dict(torch.load(path))

    # Create data
    training_seqs = conf.get_list('dataset.train_set')
    validation_seqs = conf.get_list('dataset.validation_set')
    test_seqs = conf.get_list('dataset.test_set')

    # Loss functions
    loss_func = getattr(loss_functions, conf.get_string('loss.func'))(conf)
    reproj_error = getattr(loss_functions, 'ESFMLoss')(conf)

    # Optimizer params
    lr = conf.get_float('train.lr')
    scheduler_milestone = conf.get_list('train.scheduler_milestone')
    gamma = conf.get_float('train.gamma', default=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestone, gamma=gamma)

    # Training params
    num_of_epochs = conf.get_int('train.num_of_epochs')
    num_epochs_per_seq = num_of_epochs // len(training_seqs)
    eval_intervals = conf.get_int('train.eval_intervals', default=500)
    validation_metric = conf.get_list('train.validation_metric', default=["our_repro"])
    best_validation_metric = math.inf
    best_epoch = 0
    best_model = torch.empty(0)
    converge_time = -1
    begin_time = time()
    no_ba_during_training = not conf.get_bool('ba.only_last_eval')

    for epoch in range(num_of_epochs):

        # Training
        for seq_num, scene_data in enumerate(SceneData.create_scene_data_from_list(training_seqs, conf)):
            # Optimize one Scene at a time.
            scene_dataset = ScenesDataSet.ScenesDataSet([scene_data], return_all=True)
            scene_loader = ScenesDataSet.DataLoader(scene_dataset).to(device)
            for _ in range(300):
                mean_train_loss, train_losses = train.epoch_train(scene_loader, model, loss_func, optimizer, scheduler, epoch)
            print('Training Sequence {} Loss: {}'.format(seq_num, mean_train_loss))

        # Validation
        for scene_data in SceneData.create_scene_data_from_list(validation_seqs, conf):
            scene_dataset = ScenesDataSet.ScenesDataSet([scene_data], return_all=True)
            scene_loader = ScenesDataSet.DataLoader(scene_dataset).to(device)
            for _ in range(100):
                mean_train_loss, train_losses = train.epoch_train(scene_loader, model, reproj_error, optimizer, scheduler, epoch)
            train_errors = train.epoch_evaluation(scene_loader, model, conf, epoch, Phases.VALIDATION, save_predictions=True, bundle_adjustment=False)

    # Testing
    for scene_data in SceneData.create_scene_data_from_list(test_seqs, conf):
            scene_dataset = ScenesDataSet.ScenesDataSet([scene_data], return_all=True)
            scene_loader = ScenesDataSet.DataLoader(scene_dataset).to(device)
            train_errors = train.epoch_evaluation(scene_loader, model, conf, None, Phases.TEST, save_predictions=True, bundle_adjustment=True)


if __name__ == "__main__":
    conf, device, phase = general_utils.init_exp(Phases.OPTIMIZATION.name)
    train_single_model(conf, device, phase)
