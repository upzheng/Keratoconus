def write_scalars(writer, scalars, names, n_iter, tag=None):

    for scalar, name in zip(scalars, names):
        if tag is not None:
            name = '/'.join([tag, name])
        writer.add_scalar(name, scalar, n_iter)


def write_hist_parameters(writer, net, n_iter):
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

