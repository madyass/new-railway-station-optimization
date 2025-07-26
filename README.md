#  New Railway Station Optimization

This project focuses on optimizing the placement of **new metro lines and stations** using **Genetic Algorithms (GA)**. The optimization is based on several criteria such as:

- Population density  
- Construction cost  
- Accessibility and coverage  
- Proximity to existing lines  

Currently, the project is tailored for **Istanbul**, but future versions will support other cities.

---

##  Optimization Approach

We use **Genetic Algorithms** as a metaheuristic approach due to their effectiveness in handling large, combinatorial search spaces. The system evaluates candidate networks and evolves them over generations to find near-optimal metro layouts.

---

##  Current Features

- Population-based station candidate selection  
- Initial metro line visualization  
- GA-based fitness evaluation for various city regions  
- Visualization of low, medium, and high-quality GA results  

---

##  Visualizations

###  Candidate Stations
<img width="1458" height="854" alt="candidates" src="https://github.com/user-attachments/assets/e3f9967d-7320-45fb-876f-08e12d7610bf" />

###  Neighborhood Distribution
<img width="1497" height="937" alt="neighborhoods" src="https://github.com/user-attachments/assets/095678cd-763c-4c0b-a3bb-a1472c05f780" />

###  Initial Metro Lines
<img width="992" height="836" alt="initial_metro_lines" src="https://github.com/user-attachments/assets/6351ee5b-60eb-438c-82dd-989b8e046820" />

###  GA Optimization Results
#### High Fitness Result
<img width="1028" height="876" alt="high_GA1" src="https://github.com/user-attachments/assets/c8eca090-8b17-432c-87c4-d54d65263b93" />

#### Normal Fitness Result
<img width="1058" height="831" alt="normal_GA4" src="https://github.com/user-attachments/assets/fbce9069-1c3e-4d09-b05a-27fc53f79a16" />

#### Low Fitness Result
<img width="1007" height="845" alt="low_GA4" src="https://github.com/user-attachments/assets/358f3d84-f58d-44b9-8b9f-d5da6dc651bf" />

---

##  Future Work

- Multi-city support  
- Integration with real-time traffic or transport data  
- GUI for user-defined constraints  
- Advanced fitness function including economic or environmental factors

---

##  Technologies Used

- Python  
- Matplotlib / Geopandas (for plotting)  
- Genetic Algorithm Framework  
- Pandas / Numpy  
- Jupyter Notebooks  

---
