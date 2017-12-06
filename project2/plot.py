import matplotlib.pyplot as plt

def plot_results(model_dir, run_time, run_name, history):

    complete_name = model_dir + run_time + '_' + run_name + '_results'

    losses = [i[0] for i in history]
    accs = [i[1] for i in history]

    num_epochs = range(1, len(history)+1)

    plt.plot(num_epochs, losses, color='r', label='Training loss')
    plt.plot(num_epochs, accs, color='b', label='Validation accuracy')
    plt.xlabel('epoch')
    plt.title('CNN')
    leg = plt.legend(loc='center right', shadow=True)
    leg.draw_frame(False)
    plt.savefig(complete_name + '.png')
    plt.gcf().clear()