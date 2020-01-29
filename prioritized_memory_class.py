import numpy as np

#value of 0 indicates an unwritten subtree
class SumTree:
    def __init__(self, capacity):
        self.data = [0 for i in range(capacity)]
        self.tree = [0 for i in range(2 * capacity)]#index 0 is unused for ease of indexing
        self.write_index = 0
        self.capacity = capacity

    def get_sum(self):
        return self.tree[1]

    #replace at write index, increment write index, propagate sum change
    #if at capacity, wrap around to 0
    def add(self, item, value):
        assert(value > 0) #we use 0 to represent unwritten subtrees. all written portions should have nonzero values

        change_index = self.capacity + self.write_index
        self.data[self.write_index] = item
        self.propagate(change_index, value - self.tree[change_index])
        self.write_index = (self.write_index + 1) % self.capacity

    #propogate sum change after adding
    def propagate(self, index, change):
        self.tree[index] += change
        if index == 1:
            return
        self.propagate(int(index / 2), change)

    #alpha controls how much the sums actually matter (alpha = 0 makes everything equally likely to be picked)
    def get_at_sum(self, s, index, alpha):
        assert(1 >= alpha >= 0)
        assert(index > 0) #for ease of indexing, index 0 isn't used
        assert(s <= self.tree[1]) #essentially prevents out of bounds

        if 2 * index >= len(self.tree):
            return self.data[index - self.capacity]

        leftsum = self.tree[2 * index]
        rightsum = self.tree[2 * index + 1]

        #this keeps the total sum the same while pulling leftsum and rightsum closer together
        adjustment = np.exp(-100 * alpha) * (rightsum - leftsum)/2
        leftsum += adjustment
        rightsum -= adjustment

        if s <= leftsum:
            return self.get_at_sum(s, 2 * index, alpha)
        else:
            return self.get_at_sum(s - self.tree[2 * index], 2 * index + 1, alpha)

    def print_tree(self):
        for n in range(int(np.log(self.capacity)/np.log(2) + 1.5)):
            start_index = 2 ** n
            for i in range(start_index, start_index * 2):
                print(str(self.tree[i]) + ' ', end='')
            print('\n')

class Memory:
    def __init__(self, max_samples):
        self.sumtree = SumTree(max_samples)

    def add_sample(self, sample, error):
        self.sumtree.add(sample, error + 0.01)

    def sample_samples(self, num_samples, alpha):
        samples = []
        treesum = self.sumtree.get_sum()
        for i in range(num_samples):
            s = np.random.random() * treesum
            samples.append(self.sumtree.get_at_sum(s, 1, alpha))
        return samples
