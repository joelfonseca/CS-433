"""Plot peformance results.""" 
import seaborn as sns
import matplotlib.pyplot as plt

def show_ridge_performance(results, degrees, lambdas, jet):
    """Shows the results for different parameters of the ridge regression for specific jet value dataset."""

    acc_means = []
    for i in range(1,len(degrees)):
        acc_means_by_degree = []
        for j in range(1, len(lambdas)):
            acc_means_by_degree.extend([x[5] for x in results if (x[0]==degrees[i] and x[4]==lambdas[j])])
        acc_means.append(acc_means_by_degree)

    lambdas = [round(x,5) for x in lambdas]
    ax = sns.heatmap(acc_means, xticklabels=lambdas, yticklabels=degrees, cmap='cubehelix', vmin=0.78, vmax=0.85)
    ax.invert_yaxis()
    ax.set_xlabel('lambda', labelpad=20)
    ax.set_ylabel('degree', labelpad=20)
    ax.set_title('Accuracy of data set where $PRI\_jet\_num$=' + jet + ' using Ridge Regression\n\n')
    plt.savefig('./report/figures/' + jet + '_ridge_regression_white_version.pdf', bbox_inches='tight')
    plt.show()