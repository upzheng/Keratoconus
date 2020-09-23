def recall(predict, target):
    predict = predict.byte()
    target = target.byte()
    return (predict & target).float().sum() / target.sum()


def precision(predict, target):
    predict = predict.byte()
    target = target.byte()
    return (predict & target).float().sum() / predict.sum()


def accuracy(predict, target):
    predict = predict.byte()
    target = target.byte()
    return (predict & target).float().sum() / target.size()[0]