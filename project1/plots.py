"""Plots peformance results.""" 
import seaborn as sns
import matplotlib.pyplot as plt

def show_ridge_performance(results, degrees, lambdas, jet):
    """Shows the results for different parameters of the ridge regression for specific jet value dataset."""

    acc_means = []
    for i in range(1,len(degrees)):
        acc_means_by_degree = []
        for j in range(1, len(lambdas)):
            acc_means_by_degree.extend([x[4] for x in results if (x[0]==degrees[i] and x[3]==lambdas[j])])
        acc_means.append(acc_means_by_degree)

    lambdas = [round(x,5) for x in lambdas]
    ax = sns.heatmap(acc_means, xticklabels=lambdas, yticklabels=degrees, cmap='cubehelix_r', vmin=0.78, vmax=0.85)
    ax.invert_yaxis()
    ax.set_xlabel('lambda', labelpad=20)
    ax.set_ylabel('degree', labelpad=20)
    ax.set_title('Accuracy of data set where $PRI\_jet\_num$=%d using Ridge Regression\n\n' % jet)
    plt.savefig('./report/figures/%d_ridge_regression.pdf' % jet, bbox_inches='tight')
    plt.show()

def show_logistic_performance(results, degrees, gammas, jet):
    """Shows the results for different parameters of the logistic regression for specific jet value dataset."""

    print('salut')
    acc_means = []
    for i in range(1,len(degrees)):
        acc_means_by_degree = []
        for j in range(1, len(gammas)):
            acc_means_by_degree.extend([x[4] for x in results if (x[0]==degrees[i] and x[2]==gammas[j])])
        acc_means.append(acc_means_by_degree)

    gammas = [round(x,5) for x in gammas]
    ax = sns.heatmap(acc_means, xticklabels=gammas, yticklabels=degrees, cmap='cubehelix_r', vmin=0.47, vmax=0.82)
    ax.invert_yaxis()
    ax.set_xlabel('gamma', labelpad=20)
    ax.set_ylabel('degree', labelpad=20)
    ax.set_title('Accuracy of data set where $PRI\_jet\_num$=%d using Logistic Regression\n\n' % jet)
    plt.savefig('./report/figures/%d_logistic_regression.pdf' % jet, bbox_inches='tight')
    plt.show()