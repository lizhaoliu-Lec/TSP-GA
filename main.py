import utils
import random
import argparse
import tsp_ga as ga
from datetime import datetime
from base import select, crossover, mutate


def main(args):
    genes = utils.get_genes_from(args.cities_file)

    if args.verbose:
        print("-- Running with {} cities --".format(len(genes)))

    GA = ga.SimpleGeneticAlgorithm(select_fn=select,
                                   crossover_fn=crossover,
                                   mutate_fn=mutate,
                                   population_size=args.pop_size,
                                   num_generation=args.n_gen,
                                   tournament_size=args.tour_size,
                                   mutation_rate=args.mr,
                                   crossover_rate=args.cr,
                                   elitism_ratio=args.er,
                                   verbose=args.verbose,
                                   print_every=args.print_every)
    GA.init(genes=genes)
    history = GA.run()

    if args.verbose:
        print("-- Drawing Route --")

    utils.plot(history['cost'], history['route'], args.save_dir)

    if args.verbose:
        print("-- Done --")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--no_verbose', dest='verbose', action='store_false',
                        help='Print out information or not')
    parser.add_argument('--cities_file', type=str, default="data/berlin52.csv",
                        help='Data containing the geographical coordinates of cities')
    parser.add_argument('--pop_size', type=int, default=200, help='Population size')
    parser.add_argument('--n_gen', type=int, default=500, help='Number of equal generations before stopping')
    parser.add_argument('--tour_size', type=int, default=2, help='Tournament size for competition')
    parser.add_argument('--mr', type=float, default=0.02, help='Mutation rate')
    parser.add_argument('--cr', type=float, default=0.8, help='Crossover rate')
    parser.add_argument('--er', type=float, default=0.5, help='Elitism keeping rate')
    parser.add_argument('--print_every', type=int, default=10, help='Interval to print cost')
    parser.add_argument('--save_dir', type=str, default='results/sample_result.png', help='Path to save result')

    random.seed(datetime.now())
    args = parser.parse_args()

    main(args)
