from random import sample
from math import floor, sqrt


class Gene(object):
    """
    City keep distances from cities saved in a table to improve execution time.
    """
    __distances_table = {}

    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    @staticmethod
    def distance(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x1 - x2, y1 - y2
        return floor(0.5 + sqrt(dx * dx + dy * dy))

    def get_distance_to(self, dest):
        origin = (self.x, self.y)
        dest = (dest.x, dest.y)

        forward_key = origin + dest

        if forward_key in Gene.__distances_table:
            return Gene.__distances_table[forward_key]

        dist = self.distance(origin, dest)
        Gene.__distances_table[forward_key] = dist

        return dist


class Individual(object):
    """
    possible solution to TSP
    """

    def __init__(self, genes):
        assert (len(genes) > 3)
        self.genes = genes
        self.num_genes = len(genes)
        self.__fitness = 0
        self.__cost = 0

    def new_individual(self):
        return Individual([None] * self.num_genes)

    def swap(self, idx1, idx2):
        self.genes[idx1], self.genes[idx2] = self.genes[idx2], self.genes[idx1]
        self.__reset_params()

    @property
    def fitness(self):
        if self.__fitness == 0:
            # Normalize travel cost
            self.__fitness = 1 / self.cost
        return self.__fitness

    @fitness.setter
    def fitness(self, fitness):
        self.__fitness = fitness

    @property
    def cost(self):
        if self.__cost == 0:
            for i in range(len(self.genes)):
                origin = self.genes[i]
                if i == len(self.genes) - 1:
                    dest = self.genes[0]
                else:
                    dest = self.genes[i + 1]

                self.__cost += origin.get_distance_to(dest)
        return self.__cost

    @cost.setter
    def cost(self, cost):
        self.__fitness = cost

    def __reset_params(self):
        self.__fitness = 0
        self.__cost = 0

    def __getitem__(self, item):
        return self.genes[item]

    def __lt__(self, other):
        return self.fitness < other.fitness


class Population(object):
    """
    Population of individuals
    """

    def __init__(self, individuals):
        self.individuals = individuals

    @staticmethod
    def new_population():
        return Population([])

    @staticmethod
    def generate_population(population_size, genes):
        individuals = [Individual(sample(genes, len(genes))) for _ in range(population_size)]
        return Population(individuals)

    def add(self, individual):
        self.individuals.append(individual)

    def rmv(self, individual):
        self.individuals.remove(individual)

    def get_fittest(self):
        return max(self.individuals)

    def __getitem__(self, item):
        return self.individuals[item]

    def __len__(self):
        return len(self.individuals)
