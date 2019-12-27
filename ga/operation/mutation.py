from random import random

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


name2mutation = {
    'FlipBitMutation': FlipBitMutation,
    'InverseMutation': InverseMutation,
    'FlipInverseMutation': FlipInverseMutation,
}


def get_mutation(args):
    mutation_type = args.mutation_type
    if mutation_type not in name2mutation:
        raise ValueError('Only support mutation type: %s' ','.join(list(name2mutation.keys())))
    print('Using Mutation: %s' % mutation_type)
    Mutation = name2mutation[mutation_type]
    return Mutation(args.mr)
