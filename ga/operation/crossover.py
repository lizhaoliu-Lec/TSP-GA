from random import random

from utils.func import get_k_idx


class GACrossover(object):
    """
    Class for providing an interface to easily extend the behavior of crossover
    operation between two individuals for children breeding.
    """

    def __init__(self, cross_rate=0.8):
        self.cross_rate = cross_rate

    def cross(self, father, mother):
        """
        Called when we need to cross parents to generate children.
        """
        raise NotImplementedError


class TwoPointOrderedCrossover(GACrossover):
    def cross(self, father, mother):
        if random() < self.cross_rate:
            num_genes = len(father.genes)
            # get the interval
            start_at, finish_at = get_k_idx(0, num_genes, k=2, sort=True)
            child1 = father.new_individual()
            child2 = mother.new_individual()
            child1.genes[start_at:finish_at] = mother.genes[start_at:finish_at]
            child2.genes[start_at:finish_at] = father.genes[start_at:finish_at]

            temp1 = [g for g in father.genes if g not in child1.genes]
            temp2 = [g for g in mother.genes if g not in child2.genes]
            cnt1, cnt2 = 0, 0
            t1, t2 = len(temp1), len(temp2)
            for i in range(num_genes):
                if child1.genes[i] is None:
                    child1.genes[i] = temp1[cnt1]
                    cnt1 += 1
                if child2.genes[i] is None:
                    child2.genes[i] = temp2[cnt2]
                    cnt2 += 1
                if t1 == cnt1 and t2 == cnt2:
                    break
            return child1, child2
        else:
            return father, mother


name2crossover = {
    'TwoPointOrderedCrossover': TwoPointOrderedCrossover
}


def get_crossover(args):
    crossover_type = args.crossover_type
    if crossover_type not in name2crossover:
        raise ValueError('Only support crossover type: %s' ','.join(list(name2crossover.keys())))
    print('Using Crossover: %s' % crossover_type)
    Crossover = name2crossover[crossover_type]
    return Crossover(args.cr)
