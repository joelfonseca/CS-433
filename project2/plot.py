import matplotlib.pyplot as plt

def plot_results(model_dir, run_time, run_name, loss_acc_validation_track, loss_training_track):

    complete_name = model_dir + run_time + '_' + run_name + '_results'

    losses = [i[0] for i in loss_acc_validation_track]
    accs = [i[1] for i in loss_acc_validation_track]

    num_epochs = range(1, len(loss_acc_validation_track)+1)

    plt.plot(num_epochs, loss_training_track, color ='g', label='Training loss')
    plt.plot(num_epochs, losses, color='r', label='Validation loss')
    plt.plot(num_epochs, accs, color='b', label='micro-average F1-Score')
    plt.xlabel('epoch')
    plt.title('CNN')
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig(complete_name + '.png')
    plt.gcf().clear()
