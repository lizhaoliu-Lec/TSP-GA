from random import random
from math import sqrt

from utils.func import get_k_idx


class GAMutation(object):
    """
    Class for providing an interface to easily extend the behavior of mutation operation.
    """

    def __init__(self, mutation_rate=0.02):
        self.mutation_rate = mutation_rate

    def mutate(self, individual):
        """
        Called when an individual to be mutated.
        """
        raise NotImplementedError


class FlipBitMutation(GAMutation):

    def mutate(self, individual):
        num_genes = len(individual.genes)
        mutator_t = self.mutation_rate * num_genes
        mutator_i = 0
        while mutator_i < mutator_t:
            idx1, idx2 = get_k_idx(0, max_val=num_genes)
            individual.swap(idx1, idx2)
            mutator_i += 4


class InverseMutation(GAMutation):

    def mutate(self, individual):
        num_genes = len(individual.genes)
        mutator_t = self.mutation_rate * num_genes
        mutator_i = 0
        while mutator_i < mutator_t:
            start_at, finish_at = get_k_idx(0, num_genes, k=2, sort=True)
            while start_at < finish_at:
                t = individual.genes[start_at]
                individual.genes[start_at] = individual.genes[finish_at]
                individual.genes[finish_at] = t
                start_at += 1
                finish_at -= 1
            mutator_i += 2


class FlipInverseMutation(GAMutation):
    def mutate(self, individual):
        num_genes = len(individual.genes)
        mutator_t = self.mutation_rate * num_genes
        mutator_i = 0
        while mutator_i < mutator_t:
            if random() < 0.5:
                idx1, idx2 = get_k_idx(0, max_val=num_genes)
                individual.swap(idx1, idx2)
                mutator_i += 4
            else:
                start_at, finish_at = get_k_idx(0, num_genes, k=2, sort=True)
                while start_at < finish_at:
                    t = individual.genes[start_at]
                    individual.genes[start_at] = individual.genes[finish_at]
                    individual.genes[finish_at] = t
                    start_at += 1
                    finish_at -= 1
                mutator_i += 2


class WarmUpFlipInverseMutation(GAMutation):
    def __init__(self, mutation_rate, warm_up=800, decay=1000):
        super().__init__(mutation_rate)
        self.warm_up = warm_up
        self.step = 1
        self.decay = decay

    def adjust_mutation_rate(self):
        mutation_rate = self.mutation_rate
        i = self.step / self.decay
        warm_up = self.warm_up
        self.mutation_rate = mutation_rate * min(1. / sqrt(i), i / (warm_up * sqrt(warm_up)))

    def mutate(self, individual):
        num_genes = len(individual.genes)
        mutator_t = self.mutation_rate * num_genes
        mutator_i = 0
        while mutator_i < mutator_t:
            if random() < 0.5:
                idx1, idx2 = get_k_idx(0, max_val=num_genes)
                individual.swap(idx1, idx2)
                mutator_i += 4
            else:
                start_at, finish_at = get_k_idx(0, num_genes, k=2, sort=True)
                while start_at < finish_at:
                    t = individual.genes[start_at]
                    individual.genes[start_at] = individual.genes[finish_at]
                    individual.genes[finish_at] = t
                    start_at += 1
                    finish_at -= 1
                mutator_i += 2
        self.step += 1


name2mutation = {
    'FlipBitMutation': FlipBitMutation,
    'InverseMutation': InverseMutation,
    'FlipInverseMutation': FlipInverseMutation,
    'WarmUpFlipInverseMutation': WarmUpFlipInverseMutation,
}


def get_mutation(args):
    mutation_type = args.mutation_type
    if mutation_type not in name2mutation:
        raise ValueError('Only support mutation type: %s' ','.join(list(name2mutation.keys())))
    print('Using Mutation: %s' % mutation_type)
    Mutation = name2mutation[mutation_type]
    if mutation_type == 'WarmUpFlipInverseMutation':
        return Mutation(mutation_rate=args.mr, warm_up=args.warm_up, decay=args.decay)
    return Mutation(args.mr)
