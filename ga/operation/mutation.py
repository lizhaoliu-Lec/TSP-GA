from random import random, shuffle
from copy import deepcopy
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
    def __init__(self, mutation_rate, warm_up=800, base=5):
        super().__init__(mutation_rate)
        self.warm_up = warm_up
        self.step = 1
        self.base = base

    def adjust_mutation_rate(self):
        i = self.step
        warm_up = self.warm_up
        base = self.base
        self.mutation_rate = base * min(1. / sqrt(i), i / (warm_up * sqrt(warm_up)))

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
        self.adjust_mutation_rate()
        self.step += 1


class InsertMutation(GAMutation):

    def mutate(self, individual):
        num_genes = len(individual.genes)
        mutator_t = self.mutation_rate * num_genes
        mutator_i = 0
        while mutator_i < mutator_t:
            start_at, finish_at = get_k_idx(0, num_genes, k=2, sort=True)
            fraction_1 = [g for g in individual.genes[:start_at + 1]]
            fraction_2 = [individual.genes[finish_at]]
            fraction_3 = [g for g in individual.genes[start_at + 1:finish_at]]
            fraction_4 = [g for g in individual.genes[finish_at + 1:]]
            cnt = 0
            for g in fraction_1:
                individual.genes[cnt] = g
                cnt += 1
            for g in fraction_2:
                individual.genes[cnt] = g
                cnt += 1
            for g in fraction_3:
                individual.genes[cnt] = g
                cnt += 1
            for g in fraction_4:
                individual.genes[cnt] = g
                cnt += 1
            mutator_i += 2


class ScrambleMutation(GAMutation):
    def mutate(self, individual):
        num_genes = len(individual.genes)
        mutator_t = self.mutation_rate * num_genes
        mutator_i = 0
        while mutator_i < mutator_t:
            start_at, finish_at = get_k_idx(0, num_genes, k=2, sort=True)
            rand_idx = [i for i in range(start_at, finish_at + 1)]
            shuffle(rand_idx)
            old_genes = [g for g in individual.genes[start_at:finish_at]]
            cnt = 0
            for i in range(start_at, finish_at):
                individual.genes[i] = old_genes[cnt]
                cnt += 1
            mutator_i += 2


name2mutation = {
    'FlipBitMutation': FlipBitMutation,
    'InverseMutation': InverseMutation,
    'FlipInverseMutation': FlipInverseMutation,
    'WarmUpFlipInverseMutation': WarmUpFlipInverseMutation,
    'InsertMutation': InsertMutation,
    'ScrambleMutation': ScrambleMutation,
}


def get_mutation(args):
    mutation_type = args.mutation_type
    if mutation_type not in name2mutation:
        raise ValueError('Only support mutation type: %s' ','.join(list(name2mutation.keys())))
    print('Using Mutation: %s' % mutation_type)
    Mutation = name2mutation[mutation_type]
    if mutation_type == 'WarmUpFlipInverseMutation':
        return Mutation(mutation_rate=args.mr, warm_up=args.m_warm_up, base=args.m_base)
    return Mutation(args.mr)
