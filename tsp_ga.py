from random import random, shuffle
from sys import maxsize
from time import time

from base import Population


class SimpleGeneticAlgorithm(object):
    """
    Simple Genetic algorithm
    """

    def __init__(self,
                 select_fn, crossover_fn, mutate_fn,
                 population_size, num_generation,
                 tournament_size,
                 mutation_rate=0.02, crossover_rate=0.8,
                 elitism_size=None, elitism_ratio=None,
                 verbose=True, print_every=10,
                 *args, **kwargs):
        self.select_fn = select_fn
        self.crossover_fn = crossover_fn
        self.mutate_fn = mutate_fn
        self.population_size = population_size
        self.num_generation = num_generation
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        assert elitism_size or elitism_ratio, 'should provide one of elitism_size or elitism_ratio'
        self.elitism_size = elitism_size or elitism_ratio * self.population_size
        self.elitism_size = int(self.elitism_size)
        self.tournament_size = tournament_size
        assert self.tournament_size < self.population_size, 'tournament_size should smaller that population_size'
        assert self.elitism_size < self.population_size, 'elitism_size should smaller that population_size'

        self.verbose = verbose
        self.print_every = print_every
        self.population = None
        self.history = {}

    def init(self, genes):
        if self.verbose:
            print("-- TSP-GA -- Initiating...")
        self.population = Population.generate_population(self.population_size, genes)
        self.history = {'cost': [self.population.get_fittest().travel_cost]}
        if self.verbose:
            print("-- TSP-GA -- Done initiating...")

    def evolve(self):
        new_population = Population([])
        pop_size = self.population_size
        elitism_size = self.elitism_size
        tour_size = self.tournament_size
        pop = self.population

        # Elitism
        for _ in range(elitism_size):
            fittest = pop.get_fittest()
            new_population.add(fittest)
            pop.rmv(fittest)

        # Crossover
        for _ in range(elitism_size, pop_size):
            parent_1 = self.select_fn(new_population, tour_size)
            parent_2 = self.select_fn(new_population, tour_size)
            if random() < self.crossover_rate:
                child = self.crossover_fn(parent_1, parent_2)
                new_population.add(child)
            else:
                parents = [parent_1, parent_2]
                shuffle(parents)
                new_population.add(parents[0])

        # Mutation
        for i in range(elitism_size, pop_size):
            self.mutate_fn(new_population.individuals[i], self.mutation_rate)

        self.population = new_population

    def run(self):
        min_cost = maxsize
        start_time = time()
        print("-- TSP-GA -- Running...")
        for i in range(self.num_generation):
            self.evolve()
            cost = self.population.get_fittest().travel_cost
            self.history['cost'].append(cost)
            if cost < min_cost:
                min_cost = cost
                self.history['min_cost'] = min_cost

            if self.verbose:
                if i % self.print_every == 0 or i == 0:
                    print('(%4d / %4d) current cost: %.4f, min cost: %.4f' % (i, self.num_generation, cost, min_cost))

        print("-- TSP-GA -- Done running...")

        total_time = round(time() - start_time, 6)

        if self.verbose:
            print(
                "-- TSP-GA -- Evolution finished after {} generations in {} s".format(self.num_generation, total_time))
            print("-- TSP-GA -- Minimum travelling cost {}".format(self.history['min_cost']))

        self.history['generations'] = self.num_generation
        self.history['total_time'] = total_time
        self.history['route'] = self.population.get_fittest()

        return self.history
