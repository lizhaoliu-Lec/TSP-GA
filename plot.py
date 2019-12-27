import matplotlib.pyplot as plt


def plot_summary(costs, individual, save_to=None):
    plt.rcParams['figure.figsize'] = (6.0, 3.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.figure(1)
    plt.subplot(121)
    plot_ga_convergence(costs)

    plt.subplot(122)
    plot_route(individual)
    plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, dpi=300)
    plt.show()


def plot_ga_convergence(costs):
    x = range(len(costs))
    plt.title("GA Convergence")
    plt.xlabel('num generation')
    plt.ylabel('cost')
    plt.text(x[len(x) // 2], costs[0], 'min cost: {}'.format(min(costs)), ha='center', va='center')
    plt.plot(x, costs, '-')


def plot_route(individual):
    plt.axis('off')
    plt.title("Shortest Route")

    for i in range(0, len(individual.genes)):
        x, y = individual.genes[i].x, individual.genes[i].y

        plt.plot(x, y, 'ok', c='g', markersize=5)
        if i == len(individual.genes) - 1:
            x2, y2 = individual.genes[0].x, individual.genes[0].y
        else:
            x2, y2 = individual.genes[i + 1].x, individual.genes[i + 1].y

        plt.plot([x, x2], [y, y2], 'k-', c='r')
