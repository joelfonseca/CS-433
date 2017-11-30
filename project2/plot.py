import matplotlib.pyplot as plt

def plot_results(model_name, loss_acc_track):

    losses = [i[0] for i in loss_acc_track]
    accs = [i[1] for i in loss_acc_track]

    num_epochs = range(1, len(loss_acc_track)+1)

    plt.plot(num_epochs, losses, color='r', label='Loss')
    plt.plot(num_epochs, accs, color='b', label='micro-average F1-Score')
    plt.xlabel('epoch')
    plt.title('CNN')
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig(model_name + '_results.png')
    plt.gcf().clear()