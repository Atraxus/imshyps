class HyperParameter:
    name: str
    samples: list
    default: float
    sample_idx: int

    def __init__(self, name: str, samples: list, default: float):
        self.name = name
        self.samples = samples
        self.default = default
        self.type = type
        self.sample_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.sample_idx < len(self.samples):
            sample = self.samples[self.sample_idx]
            self.sample_idx += 1
            return sample
        else:
            raise StopIteration
