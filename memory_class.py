import random

class Memory:
    def __init__(self, max_samples):
        self.samples = []
        self.max_samples = max_samples

    def add_sample(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

    def sample_samples(self, num_samples):
        if num_samples > len(self.samples):
            return self.samples
        else:
            return random.sample(self.samples, num_samples)
