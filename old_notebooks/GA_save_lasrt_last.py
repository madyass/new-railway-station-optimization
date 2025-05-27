import numpy as np
import random
import pandas as pd
from functions import haversine 
from collections import defaultdict

INITIAL_POP = 2000000 #initial population to calculate increase of population

class GeneticMetroPlanner:
    """
    neighborhood_df : DataFrame which includes all neighborr ids , coordinates and candidate stations which could be arrive neighboors
    all_stations_df : DataFrame which includes all station ids , populations and coordinates
    connectivity_dict : Dictionary which shows which station could be connected whichs stations
    existing_lines_dict : Dictionary which is existing metro lines and stations
    normalization_array : max values for veriables of fitness function to normalize(min-max)
    alpha : tunable weight for the population/cost
    w3 : tunable weight for the number of transfer 
    """
    def __init__(self, all_stations_df : pd.DataFrame, 
                 neighborhood_df : pd.DataFrame ,
                 connectivity_dict : dict , 
                 existing_lines_dict : dict,
                 center_dict : dict , 
                 mutation_rate = 0.1 , mutation_line_rate = 0.1 , 
                 mutation_new_line_protect_rate = 0.8,
                 generation_number = 20, child_number = 10, selection_rate = 0.5 ,
                 max_per_station = 2 ,max_cost = 100 , 
                 random_seed = 44 ,
                 normalization_array = [1000000 , 111 , 50 , 26],
                 w1 = 1 , w2 = 4 , w3 = 1 , w4 = 1,
                 verbose = False):

        self.stations_df = all_stations_df
        self.neighborhood_df = neighborhood_df
        self.connectivity_dict = connectivity_dict
        self.existing_lines_dict = existing_lines_dict
        self.center_dict = center_dict

        self.mutation_rate = mutation_rate
        self.mutation_line_rate = mutation_line_rate
        self.mutation_new_line_protect_rate = mutation_new_line_protect_rate

        self.generation_number = generation_number
        self.child_number = child_number
        self.selection_number = int(self.child_number / selection_rate)

        self.max_per_station = max_per_station
        self.max_cost = max_cost
        self.min_population_size = int(self.child_number * 0.2)

        self.normalization_array = normalization_array

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

        self.random_seed = random_seed
        self.verbose = verbose

        self.line_count = 12

        self.candidate_station_ids = all_stations_df[
            all_stations_df['TYPE'] == 'candidate'
        ]['station_id'].tolist()

        #inital metro lines and stations
        self.number_initial_line_stations = {line_name : len(line) for line_name , line in self.existing_lines_dict.items()}
        
        self.population = []
        self.fitness_values = []
        self.cost_values = []

        self.temp_chromosome = None
        self.temp_result = None

        self.best_final_result = None
        self.best_result_all_time = None
        self.best_chromosome = None
        self.best_chromosome_all_time = None
        
        self.stop = False
        self.history = []

    def build_neighborhood_to_station_map(self):
        neighborhood_to_station = defaultdict(set)

        for _, row in self.stations_df.iterrows():
            station_id = row['station_id']
            for neighborhood_id in row['arrived_neighborhoods']:
                neighborhood_to_station[neighborhood_id].add(station_id)
        
        return neighborhood_to_station

    def add_metro_stations(self , chromosome):
        new_chromosome = {}
        avaliable_candidates = self.candidate_station_ids.copy()
        random.shuffle(avaliable_candidates)

        for line_name , line in chromosome.items():
            added_stations = []
            current_line = line.copy()

            num_new_stations = random.randint(0 , self.max_per_station)

            for _ in range(num_new_stations):

                if not current_line:
                    break
            
                last_station = current_line[-1]

                neighbors = self.connectivity_dict.get(last_station , [])

                valid_extensions = [
                    s for s in neighbors if s in avaliable_candidates and s not in current_line and s not in added_stations
                ]

                if valid_extensions:
                    new_station = random.choice(valid_extensions)
                    current_line.append(new_station)
                    added_stations.append(new_station)
                
                else:
                    break

            new_chromosome[line_name] = current_line

        return new_chromosome
    
    def generate_initial_population(self):
        """
        create intial population
        """
        for i in range(self.child_number):
            self.population.append(self.add_metro_stations(self.existing_lines_dict))

    def calculate_arrived_station_for_neighborhood(self, chromosome):
        all_stations = set(station for line in chromosome.values() for station in line)
        neighborhood_to_station = self.build_neighborhood_to_station_map()

        neighborhood_counts = {}

        for neighborhood_id in self.neighborhood_df['neighborhood_id']:
            count = len(all_stations.intersection(neighborhood_to_station.get(neighborhood_id, set())))
            if count > 0:
                neighborhood_counts[neighborhood_id] = count

        return neighborhood_counts

    def calculate_population_for_chromosome(self, chromosome , subtract_initial_pop = True):
        total_population = 0
        used_station_ids = set()
        neighborhood_counts = self.calculate_arrived_station_for_neighborhood(chromosome)

        for line_stations in chromosome.values():
            for station_id in line_stations:
                if station_id not in used_station_ids:
                    row = self.stations_df[self.stations_df['station_id'] == station_id]
                    if not row.empty:
                        station_row = row.iloc[0]
                        neighbors = station_row['arrived_neighborhoods']

                        for neighbor in neighbors:
                            pop_row = self.neighborhood_df[self.neighborhood_df['neighborhood_id'] == neighbor]
                            if not pop_row.empty:
                                pop = pop_row.iloc[0]['population']

                                if neighbor not in neighborhood_counts:
                                    total_population += pop
                                else:
                                    total_population += pop / neighborhood_counts[neighbor]
                    used_station_ids.add(station_id)

        return total_population - INITIAL_POP
    
    def calculate_cost_per_chromosome(self , chromosome):
        """
        calculate total cost for a chromosome (number of stations which are added)
        """

        total_cost = 0
        used_station_ids = set()
        existing_ids = {s for lst in self.existing_lines_dict.values() for s in lst}
        
        for line_stations in chromosome.values():
            for station_id in line_stations:
                if station_id not in used_station_ids and station_id not in existing_ids:
                    total_cost += 1
        
        return total_cost

    def calculate_transfer_number(self , chromosome , distance = 0.3):
        """
        calculates the number of tranfer station in a chromosome
        A transfer station is defines as a station that appear in more than one line
        """
        transfer_pairs = set()
        station_coords = self.stations_df.set_index("station_id")[["lat", "lon"]].to_dict("index")

        line_names = list(chromosome.keys())

        for i in range(len(line_names)):
            line1 = line_names[i]
            stations1 = chromosome[line1]

            for j in range(i + 1, len(line_names)):
                line2 = line_names[j]
                stations2 = chromosome[line2]

                for s1 in stations1:
                    for s2 in stations2:
                        if s1 == s2:
                            continue  # Zaten aynı istasyon ID'si varsa eski yöntem işlesin diye

                        if s1 not in station_coords or s2 not in station_coords:
                            continue
                        
                        coord1 = station_coords[s1]
                        coord2 = station_coords[s2]

                        dist = haversine(coord1["lat"], coord1["lon"], coord2["lat"], coord2["lon"])
                        if dist <= distance:
                            # Transfer var. Aynı iki istasyonun sıralamasız çift olarak sadece bir kez sayılmasını sağla
                            transfer_pairs.add(frozenset([s1, s2]))

        return len(transfer_pairs)
    
    def _normalize(self , values , max_value , min_value = 0  , inverse = False):
        min_val, max_val = min_value , max_value
        if max_val == min_val:
            return [1.0 for _ in values]
        
        if inverse:  # Maliyet için ters normalizasyon
            return [(max_val - v) / (max_val - min_val) for v in values]
        else:
            return [(v - min_val) / (max_val - min_val) for v in values]

    def calculate_center_stations(self , chromosome):
        
        current_stations = set()
        for line_name , line in chromosome.items():
            for station in line:
                if station not in current_stations:
                    current_stations.add(station)
        
        result = 0

        for center_id , station_ids in self.center_dict.items():
            for station in station_ids:
                if station in current_stations:
                    result += 1
                    break
        
        return result

    def fitness(self):
        """
        Calculate fitness by first normalizing population and cost separately,
        then combining them with weights w1 and w2.
        """
        # Calculate raw populations and costs for all chromosomes
        raw_populations = [self.calculate_population_for_chromosome(chrom) for chrom in self.population]
        raw_costs = [self.calculate_cost_per_chromosome(chrom) for chrom in self.population]
        raw_transfer = [self.calculate_transfer_number(chrom) for chrom in self.population]
        raw_centers = [self.calculate_center_stations(chrom) for chrom in self.population]

        # Normalize population (higher is better)
        norm_pops = self._normalize(raw_populations,max_value=self.normalization_array[0] , inverse=False)  
        norm_costs = self._normalize(raw_costs,max_value=self.normalization_array[1],  inverse=False)       
        norm_transfer = self._normalize(raw_transfer,max_value=self.normalization_array[2],  inverse=False)
        norm_centers = self._normalize(raw_centers,max_value=self.normalization_array[3],  inverse=False)
        
        # Calculate final fitness: weighted sum of normalized values
        self.fitness_values = [
            ((norm_pop)**self.w1) * (1 / (norm_cost + 1e-6))**self.w2 + self.w3 * np.log(norm_transfer) + self.w4 * norm_center
            for norm_pop, norm_cost , norm_transfer , norm_center in zip(norm_pops, norm_costs , norm_transfer , norm_centers)
        ]
    
    def best_result(self):
        """
        returns best solution
        """
        best_idx = self.fitness_values.index(max(self.fitness_values))
        best_chromosome = self.population[best_idx]
        best_score = {'population' : self.calculate_population_for_chromosome(best_chromosome),
                      'cost' : self.calculate_cost_per_chromosome(best_chromosome) , 
                      'transfer' : self.calculate_transfer_number(best_chromosome),
                        'fitness' : max(self.fitness_values),
                        'center' : self.calculate_center_stations(best_chromosome)
        }
        return best_chromosome, best_score
    
    def details(self):
        
        initial_line_number = len(self.existing_lines_dict.keys())
        final_line_number = len(self.best_chromosome.keys())

        initial_stations = set()

        for line_name , line in self.existing_lines_dict.items():
            for station in line:
                if station not in initial_stations:
                    initial_stations.add(station)
        
        final_stations = set()
        for line_name , line in self.best_chromosome.items():
            for station in line:
                if station not in final_stations:
                    final_stations.add(station)        
        
        print(f"Population : {self.best_final_result['population']}")
        print(f"Cost : {self.best_final_result['cost']}")
        print(f"The number of Center : {self.best_final_result['center']}")
        print(f"The number of transfer stations : {self.best_final_result['transfer']}")
        print(f"The number of initial metro lines : {initial_line_number} | The number of result metro lines : {final_line_number} ")
        print(f"The number of initial metro stations : {len(initial_stations)} | The number of result metro stations : {len(final_stations)}")
    
    def roulette_wheel_selection(self):
        """
        selection method , calculates probabilities for each chromosome based on their fitness values
        then select k chromosome randomly with weight
        """
        total_fitness = sum(self.fitness_values)

        selection_probs = [f / total_fitness for f in self.fitness_values]

        selected_parents = random.choices(
            population = self.population,
            weights = selection_probs,
            k = self.selection_number)

        return selected_parents
    
    def crossover(self , parent1 , parent2):
        """
        select metro lines from parents randomly. for example:
        { M1 , M2 , M3 , M4 ... }
        M1 from parent1 , M2 from parent1 , M3 from parent2 , M4 from parent1
        """
        child = {}
        all_line_names = set(parent1.keys()) | set(parent2.keys())

        for line_name in all_line_names:
            if line_name in parent1 and line_name in parent2:
                if random.random() < 0.5:
                    child[line_name] = parent1[line_name].copy()
                else:
                    child[line_name] = parent2[line_name].copy()
            elif line_name in parent1 and random.random() < self.mutation_new_line_protect_rate:
                child[line_name] = parent1[line_name].copy()
            elif line_name in parent2 and random.random() < self.mutation_new_line_protect_rate:
                child[line_name] = parent2[line_name].copy()
            
        return child
    
    def mutate(self, chromosome):
        """
        mutation adds new station at the end or remove last station randomly
        """
        
        for line_name in chromosome:
            if random.random() < self.mutation_rate: #mutation is active
                
                current_line = chromosome[line_name] 
                used_stations = {s for line in chromosome.values() for s in line}
    
               
            
                if line_name in self.number_initial_line_stations.keys():

                    if len(current_line) > self.number_initial_line_stations[line_name]: 
                        chromosome[line_name] = current_line[:-1]
                    
                else:
                    chromosome[line_name] = current_line[:-1]
                
            
        if random.random() < self.mutation_line_rate:

            candidate_line_name = random.choice(list(chromosome.keys()))
            candidate_station = random.choice(chromosome[candidate_line_name])

            self.line_count += 1
            new_line_name = f"M{self.line_count}"

            neighbors = self.connectivity_dict.get(candidate_station , [])
            valid = [s for s in neighbors if s in self.candidate_station_ids]

            if valid:
                chromosome[new_line_name] = [candidate_station]

                for _ in range(random.randint(3,5)):
                    last = chromosome[new_line_name][-1]
                    neighbors = self.connectivity_dict.get(last , [])
                    valid = [s for s in neighbors if s in self.candidate_station_ids and s not in chromosome[new_line_name]]

                    if valid:
                        chromosome[new_line_name].append(random.choice(valid))
                    else:
                        break

        return chromosome
    
    def eliminate(self):
        """Eliminates chromosomes that exceed the maximum allowed cost."""
        new_population = []
        new_fitness_values = []
        
        for chrom, fitness, cost in zip(self.population, self.fitness_values, self.cost_values):
            if cost <= self.max_cost:
                new_population.append(chrom)
                new_fitness_values.append(fitness)
        
        # Update population and fitness values
        self.population = new_population
        self.fitness_values = new_fitness_values
        
        # If population becomes too small, stop
        if len(self.population) < self.min_population_size:
            print("Population becomes too small")
            self.stop = True

    def run(self):


        random.seed(self.random_seed)
        print("Algorithm is started.")

        self.generate_initial_population()

        #algorithm begins
        for generation_number in range(self.generation_number):

            if self.stop:
                self.best_chromosome =  current_best
                self.best_final_result = current_stats
                return current_best , current_stats
                
            self.cost_values = [self.calculate_cost_per_chromosome(chrom) for chrom in self.population]

            
            #calculate fitnesses
            self.fitness()

            current_best, current_stats = self.best_result()
            self.history.append(current_stats)

            if self.best_result_all_time is None:
                self.best_result_all_time = current_stats
                self.best_chromosome_all_time = current_best

            elif current_stats['fitness'] > self.best_result_all_time['fitness']:
                self.best_chromosome_all_time = current_best
                self.best_result_all_time = current_stats

            if self.verbose:
                print("--------------------")
                print(f"Generation {generation_number+1}:")
                print(f"Best Fitness: {current_stats['fitness']:.2f}")
                print(f"Population Coverage: {current_stats['population']}")
                print(f"Cost: {current_stats['cost']}")
                print(f"Transfers: {current_stats['transfer']}")
                print(f"Center : {current_stats['center']}")
                print("--------------------\n")
            #selection 
            selected = self.roulette_wheel_selection()

            #creating new generation
            next_generation = []

            while len(next_generation) < self.child_number:
                p1, p2 = random.sample(selected, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_generation.append(child)
    
            self.population = next_generation

            self.eliminate()

        self.best_chromosome , self.best_final_result = self.best_result()
        print("Optimization completed.")

    