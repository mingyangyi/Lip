from math import ceil

import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
import torch
import torch.nn.functional as F


class Test(object):

  # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int,
                   sigma: float, device, mode='hard', beta=1.0):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device
        self.mode = mode
        self.square_sum = None
        self.ssA = 0
        self.ssB = 0
        self.beta = beta

    def test(self, x: torch.tensor, n: int, alpha: float, batch_size: int):

        if self.mode == 'both':
            logits_hard, logits_soft = self.predict(x, n, batch_size)
            predict_hard, predict_soft = torch.topk(torch.tensor(logits_hard), 2)[1], torch.topk(torch.tensor(logits_soft), 2)[1]
            scoreA_hard, scoreB_hard = [logit[index[0]] for logit, index in zip(logits_hard, predict_hard)]\
                , [logit[index[1]] for logit, index in zip(logits_hard, predict_hard)]
            scoreA_soft, scoreB_soft = [logit[index[0]] for logit, index in zip(logits_soft, predict_soft)]\
                , [logit[index[1]] for logit, index in zip(logits_soft, predict_soft)]

            self.ssA, self.ssB = self.square_sum[predict_soft[0], predict_soft[1]]
            pA_hard, pB_hard = self._lower_confidence_bound(scoreA_hard, scoreB_hard, n, alpha, 'hard')
            pA_soft, pB_soft = self._lower_confidence_bound(np.asarray(scoreA_soft), np.asarray(scoreB_soft), n, alpha, 'soft')

            output_hard = predict_hard[:, 0]
            output_hard[torch.tensor(pA_hard) < torch.tensor(pB_hard)] = Test.ABSTAIN

            output_soft = predict_soft[:, 0]
            output_soft[torch.tensor(pA_soft) < torch.tensor(pB_soft)] = Test.ABSTAIN

            return output_hard.cpu().numpy(), output_soft.cpu().numpy()

        else:
            logits = self.predict(x, n, batch_size)
            predict = torch.topk(logits.sum(0), 2)[1]
            scoreA, scoreB = [logit[index[0]] for logit, index in zip(logits, predict)]\
                , [logit[index[1]] for logit, index in zip(logits, predict)]

            if self.mode == 'soft':
                self.ssA, self.ssB = self.square_sum[predict[0], predict[1]]

            pA, pB= self._lower_confidence_bound(scoreA, scoreB, n, alpha, self.mode)

            output_hard = predict[:, 0]
            output_hard[torch.tensor(pA) < torch.tensor(pB)] = Test.ABSTAIN

            return output.cpu().numpy()

    def predict(self, x: torch.tensor, num: int, batch_size: int):
        self.base_classifier.eval()
        inputs_size = x.size()[0]
        with torch.no_grad():
            result_hard = np.zeros((inputs_size, self.num_classes), dtype=int)
            result_soft = np.zeros((inputs_size, self.num_classes), dtype=float)
            self.square_sum = np.zeros((inputs_size, self.num_classes), dtype=float)

            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((1, this_batch_size, 1, 1))
                noise = torch.randn_like(batch, device=self.device) * self.sigma

                noisy_inputs = batch + noise
                noisy_inputs = noisy_inputs.view([inputs_size * this_batch_size] + list(x[0].size()))

                predictions = self.base_classifier(noisy_inputs).view(inputs_size, this_batch_size, self.num_classes)
                predictions *= self.beta
                if self.mode == 'hard' or self.mode == 'both':
                    p_hard = predictions.argmax(2)
                    result_hard += self._count_arr(p_hard.cpu().numpy(), batch_size,
                                                     self.num_classes)

                if self.mode == 'soft' or self.mode == 'both':
                    p_soft = F.softmax(predictions, 2)
                    p_soft_square = p_soft ** 2
                    p_soft = p_soft.sum(1)
                    p_soft_square = p_soft_square.sum(1)
                    result_soft += p_soft.cpu().numpy()
                    self.square_sum += p_soft_square.cpu().numpy()

                if self.mode == 'hard':
                    return result_hard
                if self.mode == 'soft':
                    return result_soft
                else:
                    return result_hard, result_soft

    def _count_arr(self, arr: np.ndarray, batch_size: int, length: int) -> np.ndarray:

        counts = np.zeros((batch_size, length), dtype=int)
        for item_1, item_2 in zip(counts, arr):
            for idx in item_2:
                item_1[idx] += 1

        return counts

    def _lower_confidence_bound(self, NA, NB, N, alpha: float, mode):
        if mode == 'hard':
            pA_lower_bound = np.asarray([proportion_confint(item.max(), N, alpha=alpha, method="beta")[0] for item in NA])
            pB_upper_bound = np.asarray([proportion_confint(item.max(), N, alpha=alpha, method="beta")[1] for item in NB])

        else:
            sampleA_variance = (self.ssA - NA * NA / N) / (N - 1)
            sampleB_variance = (self.ssB - NB * NB / N) / (N - 1)

            sampleA_variance[sampleA_variance < 0] = 0
            sampleB_variance[sampleB_variance < 0] = 0

            t = np.log(2 / alpha)

            pA_lower_bound = NA / N - np.sqrt(2 * sampleA_variance * t / N) - 7 * t / 3 / (N - 1)
            pB_upper_bound = NB / N + np.sqrt(2 * sampleB_variance * t / N) + 7 * t / 3 / (N - 1)

        return pA_lower_bound, pB_upper_bound


def test(model, device, dataloader, num_classes, mode='hard', sigma=0.25,
         N=1, alpha=0.0005, batch=1, beta=1.0, file_path=None):
    print('===accuracy on test set(N={}, sigma={}, mode={})==='.format(N, sigma, mode))

    correct = 0
    correct_hard = 0
    correct_soft = 0
    num = 0
    model.eval()
    smooth_net = Test(model, num_classes, sigma, device, mode, beta)

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        num += len(inputs)

        if mode == 'both':
            predict_hard_this_batch, predict_soft_this_batch = smooth_net.test(inputs, N, alpha, batch)
            correct_hard_this_batch = (predict_hard_this_batch == targets.numpy()).sum().item()
            correct_soft_this_batch = (predict_soft_this_batch == targets.numpy()).sum().item()

            correct_hard += correct_hard_this_batch
            correct_soft += correct_soft_this_batch

        else:
            predict_this_batch = smooth_net.test(inputs, N, alpha, batch)
            correct_this_batch = predict_this_batch.eq(targets).sum().item()

            correct += correct_this_batch

    if mode == 'both':
        accuracy_hard, accuracy_soft = correct_hard / num, correct_soft / num

        with open(file_path, 'a') as f:
            f.writelines('accuracy_hard: {} and accuracy_soft: {}'.format(accuracy_hard, accuracy_soft))

        print('accuracy_hard: {} and accuracy_soft: {}'.format(accuracy_hard, accuracy_soft))

        return accuracy_hard, accuracy_soft
    else:
        accuracy = correct / num

        with open(file_path, 'a') as f:
            f.writelines('accuracy: {}'.format(accuracy))
        print('accuracy: {}'.format(accuracy))

        return accuracy
