# 文件3: Coach类的完整修改 (原问题中的代码)
import copy
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics

import dgcn

log = dgcn.utils.get_logger()


class Coach:

    def __init__(self, trainset, devset, testset, model, opt, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args
        self.label_to_idx = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = self.best_dev_f1, self.best_epoch, self.best_state

        # Train
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            dev_f1, dev_acc, dev_cm = self.evaluate()
            log.info("[Dev] f1: {:.4f} acc: {:.4f}".format(dev_f1, dev_acc))
            log.info("Dev Confusion Matrix:\n{}".format(dev_cm))
            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1, best_epoch, best_state = dev_f1, epoch, copy.deepcopy(self.model.state_dict())
                log.info("Saved best model.")

            test_f1, test_acc, test_cm = self.evaluate(test=True)
            log.info("[Test] f1: {:.4f} acc: {:.4f}".format(test_f1, test_acc))

        self.model.load_state_dict(best_state)
        log.info("\nBest @epoch{}:".format(best_epoch))
        dev_f1, dev_acc, _ = self.evaluate()
        test_f1, test_acc, _ = self.evaluate(test=True)
        log.info("Final Dev  f1: {:.4f} acc: {:.4f}".format(dev_f1, dev_acc))
        log.info("Final Test f1: {:.4f} acc: {:.4f}".format(test_f1, test_acc))
        return best_dev_f1, best_epoch, best_state

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()
        # for idx in tqdm(np.random.permutation(len(self.trainset)), desc="train epoch {}".format(epoch)):
        # self.trainset.shuffle()
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.trainset[idx]
            for k, v in data.items():
                data[k] = v.to('cuda' if torch.cuda.is_available() else 'cpu')
            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()

        log.info("[Epoch {}] Loss: {:.4f} Time: {:.1f}s".format(
            epoch, epoch_loss, time.time() - start_time))

    def evaluate(self, test=False):
        dataset = self.testset if test else self.devset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    data[k] = v.to('cuda' if torch.cuda.is_available() else 'cpu')
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)
            cm = metrics.confusion_matrix(golds, preds)
            # 打印混淆矩阵（以更可读的格式）
            cm_str = "\n" + "\n".join([" ".join([f"{cell:4}" for cell in row]) for row in cm])

        return f1, acc, cm_str

