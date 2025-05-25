class GeneticMetroPlanner:
    """
    all_stations_df : DataFrame which includes all station ids , populations and coordinates
    connectivity_dict : Dictionary which shows which station could be connected whichs stations
    existing_lines_dict : Dictionary which is existing metro lines and stations
    """
    def __init__(self, all_stations_df, connectivity_dict, existing_lines_dict,
                 mutation_rate = 0.1 , generation_number = 20, child_number = 10,
                 new_station_number = 30 , max_per_station = 3):

        self.stations_df = all_stations_df
        self.connectivity_dict = connectivity_dict
        self.existing_lines_dict = existing_lines_dict

        self.mutation_rate = mutation_rate
        self.generation_number = generation_number
        self.child_number = child_number

        self.new_station_number = new_station_number
        self.max_per_station = max_per_station

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

    def fitness_population(self):
        """
        calculate fitness values for each chromosome
        """
        self.fitness_values = [] 
        for chrom in self.population:
            temp = self.calculate_population_for_chromosome(chrom)
            self.fitness_values.append(temp)

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
            k = self.child_number // 2)

        return selected_parents

    def crossover(self , parent1 , parent2):
        """
        select metro lines from parents randomly. for example:
        { M1 , M2 , M3 , M4 ... }
        M1 from parent1 , M2 from parent1 , M3 from parent2 , M4 from parent1
        """
        child = {}
        for line_name in self.existing_lines_dict:
            if random.random() < 0.5:
                child[line_name] = parent1[line_name]
            else:
                child[line_name] = parent2[line_name]
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
        
        return chromosome

    def run(self):

        #initial generation
        self.generate_initial_population()

        #algorithm begins
        for generation in range(self.generation_number):

            #calculate fitnesses
            self.fitness_population()

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
    
