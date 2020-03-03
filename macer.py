import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch.nn as nn


def macer_train(method, sigma, lbd, gauss_num, beta, gamma, num_classes, model, trainloader,
                optimizer, device, label_smooth='True'):
    m = Normal(torch.tensor([0.0]).to(device),
               torch.tensor([1.0]).to(device))

    cl_total = 0.0
    rl_total = 0.0
    data_size = 0
    correct = 0

    if method == 'macer':
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            batch_size = len(inputs)
            data_size += targets.size(0)

            new_shape = [batch_size * gauss_num]
            new_shape.extend(inputs[0].shape)
            inputs = inputs.repeat((1, gauss_num, 1, 1)).view(new_shape)
            noise = torch.randn_like(inputs, device=device) * sigma

            noisy_inputs = inputs + noise

            outputs = model(noisy_inputs)

            # noise = noise.reshape([batch_size, gauss_num] + list(inputs[0].size()))
            outputs = outputs.reshape((batch_size, gauss_num, num_classes))
            if label_smooth == 'True':
                labels = label_smoothing(inputs, targets, noise, gauss_num, num_classes, device)

            # Classification loss
            if label_smooth == 'True':
                criterion = nn.KLDivLoss(size_average=False)
                outputs_logsoftmax = F.log_softmax(outputs, dim=2).mean(1)  # log_softmax
                smoothing_label = F.softmax(labels, dim=2).mean(1)
                classification_loss = criterion.forward(outputs_logsoftmax, smoothing_label)

            else:
                outputs_softmax = F.softmax(outputs, dim=2).mean(1)
                outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
                classification_loss = F.nll_loss(outputs_logsoftmax, targets, reduction='sum')

            cl_total += classification_loss.item()
            # print(classification_loss)

            # Robustness loss
            beta_outputs = outputs * beta  # only apply beta to the robustness loss
            beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)
            _, predicted = beta_outputs_softmax.max(1)
            correct += predicted.eq(targets).sum().item()

            top2 = torch.topk(beta_outputs_softmax, 2)
            top2_score = top2[0]
            top2_idx = top2[1]
            indices_correct = (top2_idx[:, 0] == targets)  # G_theta

            out0_correct, out1_correct = top2_score[indices_correct, 0], top2_score[indices_correct, 1]
            out0_correct, out1_correct = torch.clamp(out0_correct, 0, 0.9999999), torch.clamp(out1_correct, 1e-7, 1)

            robustness_loss_correct = m.icdf(out0_correct) - m.icdf(out1_correct)

            indice_1 = robustness_loss_correct <= gamma
            # indice_2 = ~(robustness_loss_correct <= gamma)

            radius_loss = (robustness_loss_correct[indice_1] * sigma).sum() / 2

            #maxmizing gradient norm for robust data
            # gradient_loss = 0
            # if len(noise[indices_correct][indice_2]) > 0:
            #     sub_noise = noise[indices_correct][indice_2]
            #     sub_outputs = F.softmax(outputs, dim=2)[indices_correct][indice_2]
            #
            #     sub_noise = sub_noise.view(sub_noise.size()[0] * gauss_num, -1)
            #     sub_outputs = sub_outputs.view(sub_outputs.size()[0] * gauss_num, -1)
            #
            #     for i in range(num_classes):
            #         gradient_loss_tmp = sub_outputs[:, i] * sub_noise[:, i] / (gauss_num * sigma ** 2)
            #         gradient_loss_tmp = (gradient_loss_tmp ** 2).sum()
            #         gradient_loss += gradient_loss_tmp

            robustness_loss = radius_loss #+ gradient_loss
            rl_total += lbd * robustness_loss.item()

            # Final objective function
            loss = classification_loss - lbd * robustness_loss
            loss /= batch_size
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        cl_total /= data_size
        rl_total /= data_size
        acc = 100 * correct / data_size

        return cl_total, rl_total, acc

    else:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model.forward(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cl_total += loss.item() * len(inputs)
            _, predicted = outputs.max(1)
            data_size += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        cl_total /= data_size
        acc = 100 * correct / data_size

        return cl_total, rl_total, acc


def label_smoothing(inputs, targets, noise, gauss_num, num_classes, device):
    #using the ratio of norm with respect to noise to smoothing label
    inputs, noise = inputs.view(inputs.size()[0], -1), noise.view(noise.size()[0], -1)
    inputs_norm_squrae, noise_norm_square = inputs.norm(2, 1).unsqueeze(1) ** 2, noise.norm(2, 1).unsqueeze(1) ** 2
    ratio = inputs_norm_squrae / (inputs_norm_squrae + noise_norm_square)

    tmp_label = torch.zeros((targets.size()[0], num_classes)).to(device)
    for i, j in zip(targets, tmp_label):
        j[i] = 1.0
    tmp_label = tmp_label.repeat(1, gauss_num)
    tmp_label = tmp_label.view(gauss_num * tmp_label.size()[0], -1)

    #smoothed label is calculated by original hard label plus identical vector
    tmp_label = ratio * tmp_label + (1 - ratio) * torch.ones_like(tmp_label)

    label = 10 * tmp_label.view(int(inputs.size()[0] / gauss_num), gauss_num, num_classes)

    return label
