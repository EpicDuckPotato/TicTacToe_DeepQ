from prioritized_memory_class import SumTree

sumtree = SumTree(8)

for i in range(8):
    sumtree.add(i + 1, i + 1)
    sumtree.print_tree()

assert(sumtree.get_at_sum(11, 1) == 5)
