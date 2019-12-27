from random import random, sample
from itertools import accumulate
from bisect import bisect_right

from ga.base import Population


class GASelection(object):
    """
    Class for providing an interface to easily extend the behavior of selection
    operation.
    """

    def __init__(self, fitness_fn):
        self.fitness_fn = fitness_fn

    def select(self, population):
        """
        Called when we need to select parents from a population to later breeding.
        """
        raise NotImplementedError


class RouletteWheelSelection(GASelection):
    """
    Selection operator with fitness proportionate selection(FPS) or
    so-called roulette-wheel selection implementation.
    """

    def select(self, population):
        """
        Select a pair of parent using FPS algorithm.
        """
        fit = [self.fitness_fn(ind) for ind in population.individuals]
        # Create roulette wheel.
        sum_fit = sum(fit)
        wheel = list(accumulate([i / sum_fit for i in fit]))
        # Select a father and a mother.
        father_idx = bisect_right(wheel, random())
        father = population[father_idx]
        mother_idx = (father_idx + 1) % len(wheel)
        mother = population[mother_idx]
        return father, mother


class TournamentSelection(GASelection):
    def __init__(self, fitness_fn, tournament_size=2):
        """
        Selection operator using Tournament Strategy with tournament size equals
        to two by default.
        """
        super().__init__(fitness_fn)
        self.tournament_size = tournament_size

    def select(self, population):
        """
        Select a pair of parent using Tournament strategy.
        """
        assert self.tournament_size < len(population), 'tournament_size should smaller that population_size'
        num_competitors = self.tournament_size
        competitors1 = Population(sample(population.individuals, num_competitors))
        competitors2 = Population(sample(population.individuals, num_competitors))

        def competition(competitors):
            return max(competitors, key=self.fitness_fn)

        father, mother = competition(competitors1), competition(competitors2)
        return father, mother


class LinearRankingSelection(GASelection):
    def __init__(self, fitness_fn, p_min=0.1, p_max=0.9):
        """
        Selection operator using Linear Ranking selection method.
        Reference: Baker J E. Adaptive selection methods for genetic
        algorithms[C]//Proceedings of an International Conference on Genetic
        Algorithms and their applications. 1985: 101-111.
        """
        # Selection probabilities for the worst and best individuals.
        self.p_min, self.p_max = p_min, p_max
        super().__init__(fitness_fn)

    def select(self, population):
        """
        Select a pair of parent individuals using linear ranking method.
        """
        # Individual number.
        NP = len(population)
        # Add rank to all individuals in population.
        sorted(population.individuals, key=self.fitness_fn, reverse=True)

        # Assign selection probabilities linearly.
        # NOTE: Here the rank i belongs to {1, ..., N}
        def p(i):
            return self.p_min + (self.p_max - self.p_min) * (i - 1) / (NP - 1)

        probabilities = [self.p_min] + [p(i) for i in range(2, NP)] + [self.p_max]
        # Normalize probabilities.
        p_sum = sum(probabilities)
        wheel = list(accumulate([p / p_sum for p in probabilities]))
        # Select parents.
        father_idx = bisect_right(wheel, random())
        father = population[father_idx]
        mother_idx = (father_idx + 1) % len(wheel)
        mother = population[mother_idx]
        return father, mother


class ExponentialRankingSelection(GASelection):
    def __init__(self, fitness_fn, base=0.5):
        """
        Selection operator using Exponential Ranking selection method.
          base: The base of exponent
          base: float in range (0.0, 1.0)
        """
        if not (0.0 < base < 1.0):
            raise ValueError('The base of exponent c must in range (0.0, 1.0)')
        self.base = base
        super().__init__(fitness_fn)

    def select(self, population):
        """
        Select a pair of parent individuals using exponential ranking method.
        """
        # Individual number.
        NP = len(population)

        # NOTE: Here the rank i belongs to {1, ..., N}
        def prob(i):
            return self.base ** (NP - i)

        probabilities = [prob(i) for i in range(1, NP + 1)]
        # Normalize probabilities.
        p_sum = sum(probabilities)
        wheel = list(accumulate([p / p_sum for p in probabilities]))
        # Select parents.
        father_idx = bisect_right(wheel, random())
        father = population[father_idx]
        mother_idx = (father_idx + 1) % len(wheel)
        mother = population[mother_idx]
        return father, mother


name2selection = {
    'RouletteWheelSelection': RouletteWheelSelection,
    'TournamentSelection': TournamentSelection,
    'LinearRankingSelection': LinearRankingSelection,
    'ExponentialRankingSelection': ExponentialRankingSelection,

}


def get_selection(fitness_fn, args):
    selection_type = args.selection_type
    if selection_type not in name2selection:
        raise ValueError('Only support selection type: %s' ','.join(list(name2selection.keys())))
    print('Using Selection: %s' % selection_type)
    Selection = name2selection[selection_type]
    if selection_type == 'RouletteWheelSelection':
        return Selection(fitness_fn)
    elif selection_type == 'TournamentSelection':
        return Selection(fitness_fn, tournament_size=args.tour_size)
    elif selection_type == 'LinearRankingSelection':
        return Selection(fitness_fn, p_min=args.p_min, p_max=args.p_max)
    elif selection_type == 'ExponentialRankingSelection':
        return Selection(fitness_fn, base=args.base)
