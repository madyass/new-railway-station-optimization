import numpy as np
import random
from functions import haversine 

class GeneticMetroPlanner:
    """
    all_stations_df : DataFrame which includes all station ids , populations and coordinates
    connectivity_dict : Dictionary which shows which station could be connected whichs stations
    existing_lines_dict : Dictionary which is existing metro lines and stations
    w1 : tunable weight for population
    w2 : tunable weight for cost
    w3 : tunable weight for the number of transfer 
    """
    def __init__(self, all_stations_df, connectivity_dict, existing_lines_dict,
                 mutation_rate = 0.1 , mutation_line_rate = 0.1 , 
                 mutation_new_line_protect_rate = 0.8,
                 generation_number = 20, child_number = 10,
                 new_station_number = 30 , max_per_station = 3 ,random_seed = 44 ,
                 w1 = 1 , w2 = 4 , w3 = 1):

        self.stations_df = all_stations_df
        self.connectivity_dict = connectivity_dict
        self.existing_lines_dict = existing_lines_dict

        self.mutation_rate = mutation_rate
        self.mutation_line_rate = mutation_line_rate
        self.mutation_new_line_protect_rate = mutation_new_line_protect_rate
        self.generation_number = generation_number
        self.child_number = child_number

        self.new_station_number = new_station_number
        self.max_per_station = max_per_station

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.random_seed = random_seed

        self.line_count = 12

        self.candidate_station_ids = all_stations_df[
            all_stations_df['TYPE'] == 'candidate'
        ]['station_id'].tolist()

        self.population = []
        self.fitness_values = []

    def generate_chromosome(self):
        """
        generate a chromosome . for each metro line add 0-5 station at the end of metro line. 
        """
        chromosome = {}
        available_candidates = self.candidate_station_ids.copy()
        random.shuffle(available_candidates)
    
        for line_name, existing_stations in self.existing_lines_dict.items():
            added_stations = []
            current_line = existing_stations.copy()
    
            num_new_stations = random.randint(0, self.max_per_station)
    
            for _ in range(num_new_stations):
                if not current_line:
                    break
    
                last_station = current_line[-1]
                neighbors = self.connectivity_dict.get(last_station, [])
    
                valid_extensions = [
                    s for s in neighbors
                    if s in available_candidates and s not in current_line and s not in added_stations
                ]
    
                if valid_extensions:
                    new_station = random.choice(valid_extensions)
                    current_line.append(new_station)
                    added_stations.append(new_station)
                    available_candidates.remove(new_station)
                else:
                    break 
    
            chromosome[line_name] = current_line
    
        return chromosome 


    def generate_initial_population(self):
        """
        create intial population
        """
        for i in range(self.child_number):
            self.population.append(self.generate_chromosome())                    


    def calculate_population_for_chromosome(self, chromosome):
        """
        calulate total accessing population for a chromosome (input)
        """
        total_population = 0
        used_station_ids = set()

        for line_stations in chromosome.values():
            for station_id in line_stations:
                if station_id not in used_station_ids:
                    row = self.stations_df[self.stations_df['station_id'] == station_id]
                    if not row.empty:
                        total_population += row.iloc[0]['arrived_population']
                        used_station_ids.add(station_id)
        
        return total_population

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
            


    def fitness(self):
        """
        Calculate fitness by first normalizing population and cost separately,
        then combining them with weights w1 and w2.
        """
        # Calculate raw populations and costs for all chromosomes
        raw_populations = [self.calculate_population_for_chromosome(chrom) for chrom in self.population]
        raw_costs = [self.calculate_cost_per_chromosome(chrom) for chrom in self.population]
        raw_transfer = [self.calculate_transfer_number(chrom) for chrom in self.population]

        # Normalize population (higher is better)
        min_pop, max_pop = min(raw_populations), max(raw_populations)
        if max_pop == min_pop:
            norm_pops = [1.0 for _ in raw_populations]
        else:
            norm_pops = [(p - min_pop) / (max_pop - min_pop) for p in raw_populations]
    
        # Normalize cost (lower is better, so we invert the normalization)
        min_cost, max_cost = min(raw_costs), max(raw_costs)
        if max_cost == min_cost:
            norm_costs = [1.0 for _ in raw_costs]
        else:
            norm_costs = [1 - (c - min_cost) / (max_cost - min_cost) for c in raw_costs]  # Inverted

        min_transfer , max_transfer = min(raw_transfer) , max(raw_transfer)
        if max_transfer == min_transfer:
            norm_transfer = [1.0 for _ in raw_transfer]
        else:
            norm_transfer = [(t - min_transfer) / (max_transfer - min_transfer) for t in raw_transfer]    
        
        # Calculate final fitness: weighted sum of normalized values
        self.fitness_values = [
            self.w1 * norm_pop + self.w2 * norm_cost + self.w3 * norm_transfer 
            for norm_pop, norm_cost , norm_transfer in zip(norm_pops, norm_costs , norm_transfer)
        ]

    def best_result(self):
        """
        returns best solution
        """
        best_idx = self.fitness_values.index(max(self.fitness_values))
        best_chromosome = self.population[best_idx]
        best_score = self.fitness_values[best_idx]
        return best_chromosome, best_score
        
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
            k = self.child_number)

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
            if line_name in parent1 and parent2:
                if random.random() < 0.5:
                    child[line_name] = parent1[line_name].copy()
                else:
                    child[line_name] = parent2[line_name].copy()
            elif line_name in parent1 and random.random() < self.mutation_new_line_protect_rate:
                child[line_name] = parent1[line_name].copy()
            elif random.random() < self.mutation_new_line_protect_rate:
                child[line_name] = parent2[line_name].copy()
            
        return child

    def mutate(self, chromosome):
        """
        mutation adds new station at the end or remove last station randomly
        """
        
        for line_name in chromosome:
            if random.random() < self.mutation_rate: #mutation is active
                mutation_type = random.choice(["add", "remove"]) #select which mutation
                current_line = chromosome[line_name] 
                used_stations = {s for line in chromosome.values() for s in line}
    
                if mutation_type == "add":
                    last_station = current_line[-1]
                    neighbors = self.connectivity_dict.get(last_station, [])
                    valid = [s for s in neighbors if s in self.candidate_station_ids and s not in used_stations] #check adding is valid
                    if valid:
                        chromosome[line_name].append(random.choice(valid)) #add
    
                elif mutation_type == "remove" and len(current_line) > len(self.existing_lines_dict[line_name]): #check removing is valid
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

    def run(self):

        random.seed(self.random_seed)
        #initial generation
        self.generate_initial_population()

        #algorithm begins
        for generation in range(self.generation_number):

            #calculate fitnesses
            self.fitness()

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

        return self.best_result()
    
