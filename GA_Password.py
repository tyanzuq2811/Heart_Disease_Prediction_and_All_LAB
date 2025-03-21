import random
import numpy as np

POPULATION_SIZE = 100
GENES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!\"#%&/()=?@${[]}"
TARGET = "LeTuanDung@28112005"

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(cls):
        return random.choice(GENES)

    @classmethod
    def create_gnome(cls):
        return [cls.mutated_genes() for _ in range(len(TARGET))]

    def mate(self, par2):
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            prob = random.random()
            if prob < 0.45:
                child_chromosome.append(gp1)
            elif prob < 0.90:
                child_chromosome.append(gp2)
            else:
                child_chromosome.append(self.mutated_genes())
        return Individual(child_chromosome)

    def cal_fitness(self):
        return sum(1 for gs, gt in zip(self.chromosome, TARGET) if gs != gt)

def main():
    global POPULATION_SIZE

    population = [Individual(Individual.create_gnome()) for _ in range(POPULATION_SIZE)]
    generation = 1
    found = False

    while not found:
        population = sorted(population, key=lambda x: x.fitness)
        if population[0].fitness == 0:
            found = True
            break

        new_generation = []
        s = int((10 * POPULATION_SIZE) / 100)
        new_generation.extend(population[:s])
        s = int((90 * POPULATION_SIZE) / 100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation

        print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")
        generation += 1

    print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")

if __name__ == '__main__':
    main()
