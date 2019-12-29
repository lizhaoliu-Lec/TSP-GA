import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from ga.base import Gene
from ga import get_ga
from ga.operation import get_selection, get_mutation, get_crossover
from plot import plot_summary


def get_genes_from(fn):
    df = pd.read_csv(fn)
    genes = [Gene(row['city'], row['x'], row['y'])
             for _, row in df.iterrows()]

    return genes


def fitness(individual):
    # Get total travelling cost
    cost = individual.cost
    if cost == 0:
        for i in range(len(individual.genes)):
            origin = individual.genes[i]
            if i == len(individual.genes) - 1:
                dest = individual.genes[0]
            else:
                dest = individual.genes[i + 1]

            cost += origin.get_distance_to(dest)

    individual.cost = cost
    individual.fitness = 1. / individual.cost
    return individual.fitness


def run(args):
    genes = get_genes_from(args.cities_file)

    if args.verbose:
        print("Running with {} cities".format(len(genes)))

    Selection = get_selection(fitness, args)

    Mutation = get_mutation(args)

    Crossover = get_crossover(args)

    GA = get_ga(args,
                selection=Selection,
                crossover=Crossover,
                mutation=Mutation,
                population_size=args.pop_size,
                num_generation=args.n_gen,
                verbose=args.verbose,
                print_every=args.print_every)

    GA.init(genes=genes)
    history = GA.run()

    if args.verbose:
        print("Drawing Route")

    plot_summary(history['cost'], history['best_individual'], args.save_dir)

    if args.verbose:
        print("Done")

    return history['best_cost']


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cities_file', type=str, default="data/berlin52.csv",
                        help='Data containing the geographical coordinates of cities')

    # General GA
    parser.add_argument('--ga_type', type=str, default='AdaptiveGeneticAlgorithm',
                        help='Specify which GA algorithm to use')
    parser.add_argument('--no_verbose', dest='verbose', action='store_false',
                        help='Print out information or not')
    parser.add_argument('--pop_size', type=int, default=200, help='Population size')
    parser.add_argument('--n_gen', type=int, default=10000, help='Number of generations before stopping')
    parser.add_argument('--print_every', type=int, default=500, help='Interval to print cost')
    parser.add_argument('--er', type=float, default=0.5, help='Elitism keeping rate')
    parser.add_argument('--d_cr', type=float, default=0.01, help="AdaptiveGeneticAlgorithm's delta crossover rate")
    parser.add_argument('--d_mr', type=float, default=0.01, help="AdaptiveGeneticAlgorithm's delta mutation rate")
    parser.add_argument('--m_cr', type=float, default=0.001, help="AdaptiveGeneticAlgorithm's min crossover rate")
    parser.add_argument('--m_mr', type=float, default=0.001, help="AdaptiveGeneticAlgorithm's min mutation rate")

    # Selection
    parser.add_argument('--selection_type', type=str, default='TournamentSelection',
                        help='Specify which selection strategy to use')
    parser.add_argument('--tour_size', type=int, default=20, help='Tournament size for competition')
    parser.add_argument('--worse_rate', type=int, default=0.05, help='Random Tournament rate for choosing worst')
    parser.add_argument('--p_min', type=float, default=0.1, help="LinearRankingSelection's min probabilities")
    parser.add_argument('--p_max', type=float, default=0.8, help="LinearRankingSelection's max probabilities")
    parser.add_argument('--base', type=float, default=0.5, help="ExponentialRankingSelection's base")

    # Mutation
    parser.add_argument('--mutation_type', type=str, default='FlipInverseMutation',
                        help='Specify which mutation strategy to use')
    parser.add_argument('--mr', type=float, default=0.5, help='Mutation rate')
    parser.add_argument('--m_warm_up', type=int, default=800, help="WarmUpFlipInverseMutation's warm up step")
    parser.add_argument('--m_base', type=float, default=5,
                        help="WarmUpFlipInverseMutation's base for mutation rate term")

    # Crossover
    parser.add_argument('--crossover_type', type=str, default='TwoPointOrderedCrossover',
                        help='Specify which crossover strategy to use')
    parser.add_argument('--cr', type=float, default=0.5, help='Crossover rate')

    # Save fig
    parser.add_argument('--save_dir', type=str, default='results/sample_result.png', help='Path to save result')

    return parser.parse_args()


def run_experiments():
    random.seed(datetime.now())
    args = args_parser()
    times = 10

    # for GA and finally work!
    # args.cities_file = 'data/berlin52.csv'
    # args.n_gen = 10000

    args.cities_file = 'data/pr76.csv'
    args.n_gen = 10000

    # args.cities_file = 'data/rat99.csv'
    # args.n_gen = 10000

    # args.cities_file = 'data/lin105.csv'
    # args.n_gen = 10000

    final_result = {'SteadyGeneticAlgorithm': [], 'AdaptiveGeneticAlgorithm': []}

    args.mr = 0.02
    args.cr = 0.8
    args.ga_type = 'SteadyGeneticAlgorithm'
    for _ in range(times):
        print('Running SteadyGeneticAlgorithm: (%d / %d)' % (_ + 1, times))
        final_result['SteadyGeneticAlgorithm'].append(run(args))

    args.mr = 0.5
    args.cr = 0.5
    args.ga_type = 'AdaptiveGeneticAlgorithm'
    for _ in range(times):
        print('Running AdaptiveGeneticAlgorithm: (%d / %d)' % (_ + 1, times))
        final_result['AdaptiveGeneticAlgorithm'].append(run(args))

    print('SteadyGeneticAlgorithm: %s' % str(final_result['SteadyGeneticAlgorithm']))
    print('SteadyGeneticAlgorithm: %.4f' % (sum(final_result['SteadyGeneticAlgorithm']) / times))
    print('AdaptiveGeneticAlgorithm: %s' % str(final_result['AdaptiveGeneticAlgorithm']))
    print('AdaptiveGeneticAlgorithm: %.4f' % (sum(final_result['AdaptiveGeneticAlgorithm']) / times))


def visualization():
    def plot_adaptive_rate(c_rates, m_rates):
        x = range(len(c_rates))
        plt.title('Adaptive probability')
        plt.xlabel('num generation')
        plt.ylabel('probability')
        plt.plot(x, c_rates, '-')
        plt.plot(x, m_rates, '-')
        plt.legend(['crossover', 'mutation'])
        plt.tight_layout()
        plt.savefig('results/vis.PNG', dpi=300)

    random.seed(datetime.now())
    args = args_parser()
    genes = get_genes_from(args.cities_file)

    if args.verbose:
        print("Running with {} cities".format(len(genes)))

    Selection = get_selection(fitness, args)

    Mutation = get_mutation(args)

    Crossover = get_crossover(args)

    GA = get_ga(args,
                selection=Selection,
                crossover=Crossover,
                mutation=Mutation,
                population_size=args.pop_size,
                num_generation=args.n_gen,
                verbose=args.verbose,
                print_every=args.print_every)

    GA.init(genes=genes)
    history = GA.run()

    if args.verbose:
        print("Drawing Route")

    plot_summary(history['cost'], history['best_individual'], args.save_dir)
    plot_adaptive_rate(GA.crossover_rate_records, GA.mutation_rate_records)

    if args.verbose:
        print("Done")


if __name__ == "__main__":
    # run_experiments()
    visualization()
