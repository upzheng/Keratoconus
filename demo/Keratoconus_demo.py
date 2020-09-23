import torch.utils.data as data_utils
import network
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score
from classification.solver import ImageSolver
from sklearn import metrics
from utils.data.dataload import Keratoconus_Dataset
from utils.data.augmentation import Kerat_Augmentation
from classification.train_Keratoconus import KeratOption
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.autograd import Variable
from pandas import DataFrame
import os
import torch


def main(args, model_path=None):
    val_data = Keratoconus_Dataset(
        img_root=args.img_root,
        data_list=args.val_csv,
        transform=Kerat_Augmentation(size=args.input_size, rescale=args.rescale, mean=args.means, std=args.stds, mode='val'),
        argu=args.Attrib,
        flip=args.OS_mirror,
        crop=args.crop,
        front=args.Attrib_F,
        back=args.Attrib_B
    )

    val_load = data_utils.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                     drop_last=False)

    torch.backends.cudnn.benchmark = False

    model = getattr(network, args.model_name.lower())(args)
    model = model.cuda()
    model.load_model(model_path)

    label_list = []
    pred_list = []
    prob_list = []
    score_all = np.zeros((0, 3))
    with torch.no_grad():
        for step, (img, label) in enumerate(val_load):
            img = Variable(img.cuda().float())
            pred, prob, score = model.predict(img, thresh=0.5)


            pred_label = pred.cpu().data.numpy().flatten()
            prob = prob.cpu().data.numpy().flatten()
            score = score.cpu().numpy()
            print('pred_label,prob', pred_label, prob)

            print('------{}/{}-------'.format(step*args.batch_size, len(val_data)))
            print('Predict: {}'. format(pred_label.tolist()))
            print('Label  : {}'.format(label.tolist()))

            label_list.extend(label.tolist())
            pred_list.extend(pred_label.tolist())
            prob_list.extend(prob.tolist())
            score_all = np.append(score_all, score, axis=0)

    label = np.array(label_list)
    pred = np.array(pred_list)
    prob = np.array(prob_list)

    # draw roc curve, add by up
    draw_roc_curve(label, score_all)

    data1 = DataFrame(pred)
    data1.to_csv('pred_label.csv')

    # metrics
    acc = accuracy_score(label, pred)
    r = recall_score(label, pred, average='macro')

    ap = precision_score(label, pred, average='macro')

    f1 = f1_score(label, pred, average='macro')

    kappa = cohen_kappa_score(label, pred)

    if args.num_classes == 2:
        TP = (label == pred) & label
        recall = (TP.sum() / label.sum())
        prec = (TP.sum() / pred.sum())
        TN = (label == pred) & (1 - label)
        specif = TN.sum() / (1 - label).sum()
        TNnR = TN.sum() / ((pred == 0).sum())
        print('recall:', recall, 'specif:', specif)
    print(prob[label != pred])

    if args.num_classes == 3:
        label_Lesion = np.array([1 if i > 0 else 0 for i in label])
        pred_Lesion = np.array([1 if j > 0 else 0 for j in pred])
        TP_Lesion = (label_Lesion == pred_Lesion) & label_Lesion
        recall_Lesion = (TP_Lesion.sum() / label_Lesion.sum())
        print('Lesion recall:', recall_Lesion)

        # for sub
        label_sub = np.array([1 if i == 1 else 0 for i in label])
        pred_sub = np.array([1 if j == 1 else 0 for j in pred])
        TP_sub = (label_sub == pred_sub) & label_sub
        recall_sub = (TP_sub.sum() / label_sub.sum())
        acc_sub = accuracy_score(label_sub, pred_sub)
        f1_sub = f1_score(label_sub, pred_sub, average='binary', pos_label=1)
        print('sub recall:{}, sub acc:{}, sub f1：{}'.format(recall_sub, acc_sub, f1_sub))

        # for kc
        label_kc = np.array([1 if i == 2 else 0 for i in label])
        pred_kc = np.array([1 if j == 2 else 0 for j in pred])
        TP_kc = (label_kc == pred_kc) & label_kc
        recall_kc = (TP_kc.sum() / label_kc.sum())
        acc_kc = accuracy_score(label_kc, pred_kc)
        f1_kc = f1_score(label_kc, pred_kc, average='binary', pos_label=1)
        print('kc recall:{}, kc acc:{}, kc f1：{}'.format(recall_kc, acc_kc, f1_kc))

    print('acc:', acc)
    print('recall:', r)
    print('prec:', ap)
    print('f1:', f1)
    print('kappa:', kappa)


def draw_roc_curve(label_all, prob_all):
    n_classes = 3
    y_all = np.zeros((label_all.shape[0], 3))
    for idx, i in enumerate(label_all):
        y_all[idx, i] = 1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_all[:, i], prob_all[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='red', lw=lw, label='ROC curve cls0 (area = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='blue', lw=lw, label='ROC curve cls1 (area = %0.2f)' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve cls2 (area = %0.2f)' % roc_auc[2])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('roc_1model.png', format='png')
    plt.close()


if __name__ == '__main__':
    options = KeratOption()
    args = options.initialize([
        'test_new',
        '--save_folder', 'weights',
        '--val_csv', 'data/Pentacam7/test_data.csv',
        '--rescale', '1.0',
        '--means', '7.9', '6.46', '1068.87', '1105.1', '622',
        '--stds', '0.66', '0.76', '654.9', '672', '80.21',
        '--input_size', '141',
        '--img_root', 'data/Pentacam7',
        '--resnet_layers', '18',
        '--model_name', 'KerNet',
        '--num_classes', '3',
        '--Attrib', 'CUR', 'ELE', 'PAC',
        '--input_dim', '5',
        '--crop', '5',
    ])
    model_root = 'weights/exp6/Resnet-18_best_recall.pth'
    main(args, model_path=model_root)
