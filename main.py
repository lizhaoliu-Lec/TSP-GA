import random
import argparse
import pandas as pd
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


def main(args):
    genes = get_genes_from(args.cities_file)

    if args.verbose:
        print("Running with {} cities".format(len(genes)))

    Selection = get_selection(fitness, args)

    Mutation = get_mutation(args)

    Crossover = get_crossover(args)

    GA = get_ga(args,
                select_fn=Selection.select,
                cross_fn=Crossover.cross,
                mutate_fn=Mutation.mutate,
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


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cities_file', type=str, default="data/berlin52.csv",
                        help='Data containing the geographical coordinates of cities')

    # General GA
    parser.add_argument('--ga_type', type=str, default='SteadyGeneticAlgorithm',
                        help='Specify which GA algorithm to use')
    parser.add_argument('--no_verbose', dest='verbose', action='store_false',
                        help='Print out information or not')
    parser.add_argument('--pop_size', type=int, default=300, help='Population size')
    parser.add_argument('--n_gen', type=int, default=500, help='Number of equal generations before stopping')
    parser.add_argument('--print_every', type=int, default=10, help='Interval to print cost')
    parser.add_argument('--er', type=float, default=0.5, help='Elitism keeping rate')

    # Selection
    parser.add_argument('--selection_type', type=str, default='TournamentSelection',
                        help='Specify which selection strategy to use')
    parser.add_argument('--tour_size', type=int, default=10, help='Tournament size for competition')
    parser.add_argument('--p_min', type=float, default=0.1, help="LinearRankingSelection's min probabilities")
    parser.add_argument('--p_max', type=float, default=0.9, help="LinearRankingSelection's max probabilities")
    parser.add_argument('--base', type=float, default=0.5, help="ExponentialRankingSelection's base")

    # Mutation
    parser.add_argument('--mutation_type', type=str, default='WarmUpFlipInverseMutation',
                        help='Specify which mutation strategy to use')
    parser.add_argument('--mr', type=float, default=0.02, help='Mutation rate')
    parser.add_argument('--warm_up', type=int, default=800, help="WarmUpFlipInverseMutation's warm up step")
    parser.add_argument('--decay', type=float, default=8000, help="WarmUpFlipInverseMutation's decay step")

    # Crossover
    parser.add_argument('--crossover_type', type=str, default='TwoPointOrderedCrossover',
                        help='Specify which crossover strategy to use')
    parser.add_argument('--cr', type=float, default=0.8, help='Crossover rate')

    # Save fig
    parser.add_argument('--save_dir', type=str, default='results/sample_result.png', help='Path to save result')

    return parser.parse_args()


if __name__ == "__main__":
    random.seed(datetime.now())
    args = args_parser()
    main(args)
