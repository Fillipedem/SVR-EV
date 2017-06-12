"""
Cuckoo Search Implementation
"""
from .optimization import Optimization
from .levy_distribution import levy_distro
import random


class CuckooSearch(Optimization):
    def __init__(self, dim, func, limits, num_nest=50, p=0.25):
        # search space
        self.dim = dim
        
        # initialize lower and upper bound
        self.__initialize_limits(limits)
        
        # fitness function
        self.func = func
        
        # Cuckoo Search values
        self.nests = []
        
        # Cuckoo Search parameter
        self.num_nest = num_nest
        self.p = p
    
    # public methods
    def search(self, max_iter):
        """
        return the best solution

        :param max_iter: max number of iteration
        :return: real vector, represeting the best solution
        """
    
        # initialize nests
        self.__initialize()
        
        # search ans
        best = self.__search(max_iter)
        
        # return best solution found
        return best
    
    
    #                 #
    ## class methods ##
    #                 #
    def __initialize_limits(self, limits):
        
        if not isinstance(limits, tuple):
            raise TypeError("args limits is not a tuple!!")
        
        if len(limits) != 2:
            raise ValueError("Len of limits is not 2!!")
        
        self.lower, self.upper = limits
    
    
    def __initialize(self):
        """
        Initialize nest
        :return: None
        """
        
        self.nests = []
        
        for _ in range(self.num_nest):
            new_egg = self.__random_solution()
            self.nests.append(new_egg)
    
    
    def __random_solution(self):
        """
        return a new random solution
        :return: real vector with size self.dim
        """
        solution = []
        
        for _ in range(self.dim):
            solution.append(random.uniform(self.lower, self.upper))
        
        return solution
    
    
    def __levy_flights(self, cuckoo):
        
        new_egg = []

        for idx, x in enumerate(cuckoo):
            step = levy_distro(2.0)
            
            while x + step > self.upper or x + step < self.lower:
                step = levy_distro(2.0)
                
            new_egg.append(x + step)
    
        return new_egg
    
    
    def __search(self, max_iter):
        """
        return the best solution
        
        :param max_iter: max number of iteration
        :return: real vector, represeting the best solution
        """
        
        # first steps is evaluated all eggs fitness
        nests_fitness = []
        
        for egg in self.nests:
            nests_fitness.append(self.func(egg))
        
        population = list(zip(self.nests, nests_fitness))
        
        # while max_iter
        while max_iter:
            # get a random cuckoo and use levy flights
            idx = random.randint(0, self.num_nest - 1)
            
            cuckoo = self.__levy_flights(population[idx][0])
            cuckoo_fitness = self.func(cuckoo)

            
            # select a random nest
            nest = random.randint(0, self.num_nest - 1)
            
            if cuckoo_fitness < population[nest][1]:
                population[nest] = cuckoo, cuckoo_fitness
            
            # order by fitness
            population.sort(key=lambda x: x[1])
            
            
            # remove the worst solutions(Nests)
            pos = int(self.num_nest - self.p*self.num_nest)
            
            for idx in range(pos, self.num_nest):
                new_egg = self.__random_solution()
                population[idx] = (new_egg, self.func(new_egg))
                
            # next ite
            max_iter -= 1
        
        # return best element found so far
        return population[0][0]


