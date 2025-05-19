import random
from collections import defaultdict

class GeneticMetroPlanner:
    def __init__(self, all_stations_df, connectivity_dict, existing_lines_dict,
                 mutation_rate = 0.1 , generation_number = 20, child_number = 10,
                 new_station_number = 30 , max_per_station = 5):

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

    def random_partition_fixed_total(self, num_bins=17):
        total = self.new_station_number
        max_per_bin = self.max_per_station

        while True:
            parts = [0] * num_bins
            for _ in range(total):
                idx = random.randint(0, num_bins - 1)
                if max_per_bin is None or parts[idx] < max_per_bin:
                    parts[idx] += 1
            if sum(parts) == total:
                return parts
                
    def generate_chromosome(self):
        chromosome = {}
        available_candidates = self.candidate_station_ids.copy()
        random.shuffle(available_candidates)

        max_new_stations_per_line = 5  # Her hat için en fazla 5 yeni istasyon

        for line_name, existing_stations in self.existing_lines_dict.items():
            added_stations = []

            for station_id in available_candidates:
                if station_id in self.connectivity_dict:
                    for existing_station in existing_stations:
                        if (existing_station in self.connectivity_dict and
                            station_id in self.connectivity_dict[existing_station]):
                            added_stations.append(station_id)
                            break  # Bağlantı bulundu, istasyonu ekle
                if len(added_stations) == max_new_stations_per_line:
                    break

            # Hat için yeni kromozom: mevcutlar + eklenenler
            chromosome[line_name] = existing_stations + added_stations

            # Eklenenleri havuzdan çıkar
            available_candidates = [s for s in available_candidates if s not in added_stations]

        return chromosome


    def generate_initial_population():
        pass