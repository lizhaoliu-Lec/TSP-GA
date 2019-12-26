from random import randint, random, sample
from math import floor, sqrt


class Gene(object):
    """
    City
    keep distances from cities saved in a table to improve execution time.
    """
    __distances_table = {}

    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def get_distance_to(self, dest):
        origin = (self.x, self.y)
        dest = (dest.x, dest.y)

        forward_key = origin + dest

        if forward_key in Gene.__distances_table:
            return Gene.__distances_table[forward_key]

        dist = distance(origin, dest)
        Gene.__distances_table[forward_key] = dist

        return dist


class Individual(object):
    """
    possible solution to TSP
    """

    def __init__(self, genes):
        assert (len(genes) > 3)
        self.genes = genes
        self.__travel_cost = 0
        self.__fitness = 0

    def swap(self, gene_1, gene_2):
        a, b = self.genes.index(gene_1), self.genes.index(gene_2)
        self.genes[b], self.genes[a] = self.genes[a], self.genes[b]
        self.__reset_params()

    def add(self, gene):
        self.genes.append(gene)
        self.__reset_params()

    @property
    def fitness(self):
        if self.__fitness == 0:
            # Normalize travel cost
            self.__fitness = 1 / self.travel_cost
        return self.__fitness

    @property
    def travel_cost(self):
        # Get total travelling cost
        if self.__travel_cost == 0:
            for i in range(len(self.genes)):
                origin = self.genes[i]
                if i == len(self.genes) - 1:
                    dest = self.genes[0]
                else:
                    dest = self.genes[i + 1]

                self.__travel_cost += origin.get_distance_to(dest)

        return self.__travel_cost

    def __reset_params(self):
        self.__travel_cost = 0
        self.__fitness = 0

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
    def generate_population(population_size, genes):
        individuals = [Individual(sample(genes, len(genes))) for _ in range(population_size)]
        return Population(individuals)

    def add(self, individual):
        self.individuals.append(individual)

    def rmv(self, individual):
        self.individuals.remove(individual)

    def get_fittest(self):
        return max(self.individuals)


def select(population, num_competitors):
    return Population(sample(population.individuals, num_competitors)).get_fittest()


def crossover(parent_1, parent_2):
    def fill_with_parent1_genes(child, parent, genes_n):
        start_at = randint(0, len(parent.genes) - genes_n - 1)
        finish_at = start_at + genes_n
        child.genes[start_at:finish_at] = parent_1.genes[start_at:finish_at]

    def fill_with_parent2_genes(child, parent):
        temp = [g for g in parent.genes if g not in child]
        count = 0
        for i in range(len(child.genes)):
            if child.genes[i] is None:
                child.genes[i] = temp[count]
                count += 1

    genes_n = len(parent_1.genes)
    child = Individual([None for _ in range(genes_n)])
    fill_with_parent1_genes(child, parent_1, genes_n // 2)
    fill_with_parent2_genes(child, parent_2)

    return child


def mutate(individual, rate=0.5):
    for _ in range(len(individual.genes)):
        if random() < rate:
            selected_genes = sample(individual.genes, 2)
            individual.swap(selected_genes[0], selected_genes[1])


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x1 - x2, y1 - y2
    return floor(0.5 + sqrt(dx * dx + dy * dy))
