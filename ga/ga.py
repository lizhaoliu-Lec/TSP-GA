from sys import maxsize
from time import time
from copy import deepcopy

from ga.base import Population


class GeneticAlgorithm(object):
    """
    Genetic algorithm interface
    """

    def __init__(self,
                 select_fn, cross_fn, mutate_fn,
                 population_size, num_generation,
                 verbose=True, print_every=10):
        self.select_fn = select_fn
        self.crossover_fn = cross_fn
        self.mutate_fn = mutate_fn
        self.population_size = population_size
        self.num_generation = num_generation

        self.verbose = verbose
        self.print_every = print_every
        self.population = None
        self.history = {}

    def init(self, genes):
        if self.verbose:
            print("Initiating...")
        self.population = Population.generate_population(self.population_size, genes)
        self.history = {'cost': [self.population.get_fittest().cost]}
        if self.verbose:
            print("Done initiating...")

    def evolve(self):
        raise NotImplemented

    def run(self):
        best_cost = maxsize
        start_time = time()
        print("Running...")
        for i in range(self.num_generation):
            self.evolve()
            cost = self.population.get_fittest().cost
            self.history['cost'].append(cost)
            if cost < best_cost:
                best_cost = cost
                self.history['best_cost'] = best_cost

            if self.verbose:
                if i % self.print_every == 0 or i == 0:
                    print('(%4d / %4d) current cost: %.2f, min cost: %.2f' % (
                        i, self.num_generation, cost, best_cost))

        print("Done running...")

        total_time = round(time() - start_time, 6)

        if self.verbose:
            print(
                "Evolution finished after {} generations in {} s".format(self.num_generation,
                                                                         total_time))
            print("Minimum travelling cost {}".format(self.history['best_cost']))

        self.history['generations'] = self.num_generation
        self.history['total_time'] = total_time
        self.history['best_individual'] = self.population.get_fittest()

        return self.history


class SimpleGeneticAlgorithm(GeneticAlgorithm):
    """
    Simple Genetic algorithm
    """

    def evolve(self):
        new_population = Population([])
        pop_size = self.population_size

        # Crossover
        for _ in range(0, pop_size):
            parent_1, parent_2 = self.select_fn(self.population)
            child1, child2 = self.crossover_fn(parent_1, parent_2)
            new_population.add(child1)
            new_population.add(child2)

        # Mutation
        for i in range(0, pop_size):
            self.mutate_fn(new_population[i])

        self.population = new_population


class ElitistReserveGeneticAlgorithm(GeneticAlgorithm):
    """
    Elitist Reserve Genetic algorithm
    """

    def __init__(self, elitism_ratio, **kwargs):
        super().__init__(**kwargs)
        self.elitism_size = elitism_ratio * self.population_size
        self.elitism_size = int(self.elitism_size)
        assert self.elitism_size < self.population_size, 'elitism_size should smaller that population_size'

    def evolve(self):
        new_population = Population([])
        pop_size = self.population_size
        elitism_size = self.elitism_size
        pop = self.population

        # Elitism
        for _ in range(elitism_size):
            fittest = pop.get_fittest()
            new_population.add(fittest)
            pop.rmv(fittest)

        # Crossover
        for _ in range(elitism_size, pop_size):
            parent_1, parent_2 = self.select_fn(new_population)
            child1, child2 = self.crossover_fn(parent_1, parent_2)
            new_population.add(child1)
            new_population.add(child2)

        # Mutation
        for i in range(elitism_size, pop_size):
            self.mutate_fn(new_population.individuals[i])

        self.population = new_population


class SteadyGeneticAlgorithm(GeneticAlgorithm):
    """
    Steady GA that choose the best two from p1, p2, c1, c2 after the crossover,
    """

    def evolve(self):
        new_population = Population([])
        pop_size = self.population_size

        # Crossover
        for _ in range(0, pop_size):
            parent_1, parent_2 = self.select_fn(self.population)
            child1, child2 = self.crossover_fn(parent_1, parent_2)
            small_group = [parent_1, parent_2, child1, child2]
            child1, child2 = sorted(small_group, reverse=True)[0:2]
            new_population.add(child1)
            new_population.add(child2)

        # Mutation
        for i in range(0, pop_size):
            self.mutate_fn(new_population[i])

        self.population = new_population


class SelectDownToSizeGeneticAlgorithm(GeneticAlgorithm):
    """
    Select Down To Size Genetic algorithm
    """

    def evolve(self):
        raise NotImplemented


class LonelyMutateGeneticAlgorithm(GeneticAlgorithm):
    """
    Lonely Mutate Genetic algorithm, No crossover will be performed
    """

    def evolve(self):
        new_population = Population([])
        pop_size = self.population_size

        # Mutation
        for i in range(0, pop_size):
            old_one = self.population[i]
            new_one = deepcopy(old_one)
            self.mutate_fn(new_one)
            if old_one.fitness >= new_one.fitness:
                new_population.add(old_one)
            else:
                new_population.add(new_one)
        self.population = new_population


name2ga = {
    'SimpleGeneticAlgorithm': SimpleGeneticAlgorithm,
    'ElitistReserveGeneticAlgorithm': ElitistReserveGeneticAlgorithm,
    'SteadyGeneticAlgorithm': SteadyGeneticAlgorithm,
    'LonelyMutateGeneticAlgorithm': LonelyMutateGeneticAlgorithm,
}


def get_ga(args, **kwargs):
    ga_type = args.ga_type
    if ga_type not in ga_type:
        raise ValueError('Only support GA type: %s' ','.join(list(name2ga.keys())))
    print('Using GA: %s' % ga_type)
    GA = name2ga[ga_type]
    if ga_type == 'ElitistReserveGeneticAlgorithm':
        return GA(elitism_ratio=args.er, **kwargs)
    elif ga_type == 'SteadyGeneticAlgorithm':
        return GA(**kwargs)
    elif ga_type == 'SimpleGeneticAlgorithm':
        return GA(**kwargs)
    elif ga_type == 'LonelyMutateGeneticAlgorithm':
        return GA(**kwargs)
