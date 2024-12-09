import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from Utilities_python import genfuel
import random

def process_generation_data(max_p_mw, gen, gencost, NB, generation = None):
    if generation is None:
        generation = pd.DataFrame()
        generation['max_p_mw'] = max_p_mw
        generation['genfuel'] = genfuel
        generation['bus'] = gen['new_bus'].reset_index(drop=True)
        generation['c'] = gencost['COST_1'].reset_index(drop=True)
    
    condition_green = generation['genfuel'].isin(["wind", "hydro", "solar", "nuclear"])
    green_generation = generation.loc[condition_green]
    
    condition_black = generation['genfuel'].isin(["coal", "ng"])
    black_generation = generation.loc[condition_black]

    map_green_generators = np.zeros((len(green_generation), NB))
    bus_GN = np.array(green_generation['bus'])
    for i in range(len(green_generation)):
        map_green_generators[i, bus_GN[i]] = 1

    map_black_generators = np.zeros((len(black_generation), NB))
    bus_GN = np.array(black_generation['bus'])
    for i in range(len(black_generation)):
        map_black_generators[i, bus_GN[i]] = 1

    max_green = np.array(green_generation['max_p_mw'].reset_index(drop=True))
    max_black = np.array(black_generation['max_p_mw'].reset_index(drop=True))
    
    return map_green_generators, map_black_generators, generation, green_generation, black_generation, max_green, max_black


def max_green_max_black(generation, ratio_green, randomseed):
    ## ratio in ercot system -> 18.67% green and 81.32% black
    # Ratio green should be between 0 and 1, a value of 0.5 would mean 50% green in the system, while a value of 0.7 would mean 70% of green generation in the system
    random.seed(randomseed)

    sum_of_gen = sum(generation['max_p_mw'])
    total_green = ratio_green*sum_of_gen

    condition_green = generation['genfuel'].isin(["wind", "hydro", "solar", "nuclear"])
    green_generation = generation.loc[condition_green]
    sum_of_green_gen = sum(green_generation['max_p_mw'])

    avg_c_ng = np.mean(generation['c'].loc[generation['genfuel'].isin(["ng"])])
    avg_c_coal = np.mean(generation['c'].loc[generation['genfuel'].isin(["coal"])])
    avg_c_nuclear = np.mean(generation['c'].loc[generation['genfuel'].isin(["nuclear"])])


    if ratio_green < sum_of_green_gen/sum_of_gen:
        while sum_of_green_gen > total_green:
            condition_black = generation['genfuel'].isin(["coal", "ng"])
            black_generation = generation.loc[condition_black]
            condition_green = generation['genfuel'].isin(["wind", "hydro", "solar", "nuclear"])
            green_generation = generation.loc[condition_green]
            # Randomly choose a row index from the filtered DataFrame
            random_index = random.choice(green_generation.index)
    
            # Assign the specific value to the chosen row in the specified column
            generation.at[random_index, 'genfuel'] = random.choice(["coal", "ng"])
            if generation.at[random_index, 'genfuel'] == 'ng':
                generation.at[random_index, 'c'] = avg_c_ng
            elif  generation.at[random_index, 'genfuel'] == 'coal':
                 generation.at[random_index, 'c'] = avg_c_coal
            condition_green = generation['genfuel'].isin(["wind", "hydro", "solar", "nuclear"])
            green_generation = generation.loc[condition_green]
            sum_of_green_gen = sum(green_generation['max_p_mw'])


    while sum_of_green_gen < total_green:
            condition_black = generation['genfuel'].isin(["coal", "ng"])
            black_generation = generation.loc[condition_black]
            # Randomly choose a row index from the filtered DataFrame
            random_index = random.choice(black_generation.index)
    
            # Assign the specific value to the chosen row in the specified column
            generation.at[random_index, 'genfuel'] = random.choice(["wind", "hydro", "solar", "nuclear"])
            if  generation.at[random_index, 'genfuel'] == 'nuclear':
                 generation.at[random_index, 'c'] = avg_c_nuclear
            else:
                generation.at[random_index, 'c'] = 0
            condition_green = generation['genfuel'].isin(["wind", "hydro", "solar", "nuclear"])
            green_generation = generation.loc[condition_green]
            sum_of_green_gen = sum(green_generation['max_p_mw'])
    return generation

def max_green_max_black_nuclear_black(generation, ratio_green, randomseed):
    ## ratio in ercot system -> 18.67% green and 81.32% black
    # Ratio green should be between 0 and 1, a value of 0.5 would mean 50% green in the system, while a value of 0.7 would mean 70% of green generation in the system
    random.seed(randomseed)

    sum_of_gen = sum(generation['max_p_mw'])
    total_green = ratio_green*sum_of_gen

    condition_green = generation['genfuel'].isin(["wind", "hydro", "solar", "nuclear"])
    green_generation = generation.loc[condition_green]
    sum_of_green_gen = sum(green_generation['max_p_mw'])

    avg_c_ng = np.mean(generation['c'].loc[generation['genfuel'].isin(["ng"])])
    avg_c_coal = np.mean(generation['c'].loc[generation['genfuel'].isin(["coal"])])
    avg_c_nuclear = np.mean(generation['c'].loc[generation['genfuel'].isin(["nuclear"])])


    if ratio_green < sum_of_green_gen/sum_of_gen:
        while sum_of_green_gen > total_green:
            condition_black = generation['genfuel'].isin(["coal", "ng", "nuclear"])
            black_generation = generation.loc[condition_black]
            condition_green = generation['genfuel'].isin(["wind", "hydro", "solar"])
            green_generation = generation.loc[condition_green]
            # Randomly choose a row index from the filtered DataFrame
            random_index = random.choice(green_generation.index)
    
            # Assign the specific value to the chosen row in the specified column
            generation.at[random_index, 'genfuel'] = random.choice(["coal", "ng", "nuclear"])
            if generation.at[random_index, 'genfuel'] == 'ng':
                generation.at[random_index, 'c'] = avg_c_ng
            elif  generation.at[random_index, 'genfuel'] == 'coal':
                 generation.at[random_index, 'c'] = avg_c_coal
            elif generation.at[random_index, 'genfuel'] == 'nuclear':
                 generation.at[random_index, 'c'] = avg_c_nuclear
            condition_green = generation['genfuel'].isin(["wind", "hydro", "solar"])
            green_generation = generation.loc[condition_green]
            sum_of_green_gen = sum(green_generation['max_p_mw'])


    while sum_of_green_gen < total_green:
            condition_black = generation['genfuel'].isin(["coal", "ng"])
            black_generation = generation.loc[condition_black]
            # Randomly choose a row index from the filtered DataFrame
            random_index = random.choice(black_generation.index)
    
            # Assign the specific value to the chosen row in the specified column
            generation.at[random_index, 'genfuel'] = random.choice(["wind", "hydro", "solar"])
            generation.at[random_index, 'c'] = 0
            condition_green = generation['genfuel'].isin(["wind", "hydro", "solar"])
            green_generation = generation.loc[condition_green]
            sum_of_green_gen = sum(green_generation['max_p_mw'])
    return generation


def fix_bus_numbers(dataframe, bus_indices, bus, bus_column_name):
    bus['bus'] = range(len(bus_indices))
    new_bus = []
    for item in dataframe[bus_column_name]:
        for idx, bus_id in enumerate(bus_indices):
            if item == bus_id:
                new_bus.append(bus['bus'][idx])
                break
    if (bus_column_name == 'F_BUS'):
        dataframe['new_from_bus'] = new_bus
    elif (bus_column_name == 'T_BUS'):
        dataframe['new_to_bus'] = new_bus
    else:
        dataframe['new_bus'] = new_bus

    return dataframe

def map_generators_to_buses(gen, new_bus, num_buses):
    map_generators = np.zeros((len(gen), num_buses))
    for i, bus in enumerate(new_bus):
        map_generators[i, bus] = 1
    return map_generators

def map_loads_to_buses(bus, baseMVA):
    loads = bus['PD'] / baseMVA
    map_loads = np.zeros((len(loads), len(loads)))
    len_loads=0
    for i, load in enumerate(loads):
        if load > 0:
            map_loads[i, i] = 1
            len_loads = len_loads+1
    map_loads = map_loads[~np.all(map_loads == 0, axis=1)]
    return map_loads, len_loads

def make_bus_susceptance_matrix(branch, baseMVA, congestion_factor, NB):
    from_bus = np.array(branch['new_from_bus'])
    to_bus = np.array(branch['new_to_bus'])
    #flow_limit = np.array(branch['RATE_A'])
    flow_limit = np.array(branch['RATE_A'])/baseMVA
    b_lines = np.array(1 / branch['BR_X'])
    parallel_lines = np.ones(len(branch))

    # Combine into a DataFrame
    df = pd.DataFrame({'from_bus': from_bus, 'to_bus': to_bus, 'flow_limit': flow_limit, 'b': b_lines, 'parallel': parallel_lines })

    # Remove duplicate rows
    # df.drop_duplicates(subset=['from_bus', 'to_bus'], keep='last', inplace=True)
    duplicate_mask = df.duplicated(subset=['from_bus', 'to_bus'], keep='last')
    df.loc[duplicate_mask, 'parallel'] = 2

    # Initialize matrices
    b = np.zeros((NB, NB))
    branch_cap = np.zeros((NB, NB))

    # Fill matrices
    for i, row in df.iterrows():
        from_bus_idx = int(row['from_bus'])  # Convert to integer index
        to_bus_idx = int(row['to_bus'])      # Convert to integer index
        b_line = row['b']
        flow_lim = row['flow_limit']

        b[from_bus_idx, to_bus_idx] = b_line
        b[to_bus_idx, from_bus_idx] = b_line
        branch_cap[from_bus_idx, to_bus_idx] = congestion_factor*flow_lim
        branch_cap[to_bus_idx, from_bus_idx] = congestion_factor*flow_lim

    # Calculate diagonal elements of b matrix
    np.fill_diagonal(b, -np.sum(b, axis=1))

    return b, branch_cap

def process_generation_data_emissions(max_p_mw, gen, gencost, NB, generation = None):
    if generation is None:
        generation = pd.DataFrame()
        generation['max_p_mw'] = max_p_mw
        generation['genfuel'] = genfuel
        generation['bus'] = gen['new_bus'].reset_index(drop=True)
        generation['c'] = gencost['COST_1'].reset_index(drop=True)
    
    condition_green = generation['genfuel'].isin(["wind", "hydro", "solar", "nuclear"])
    green_generation = generation.loc[condition_green] 

    condition_black = generation['genfuel'].isin(["coal", "ng"])
    black_generation = generation.loc[condition_black]

        
    condition_wind = generation['genfuel'].isin(["wind"])
    wind_generation = green_generation.loc[condition_wind]

    condition_coal = generation['genfuel'].isin(["coal"])
    coal_generation = black_generation.loc[condition_coal]

    condition_ng = generation['genfuel'].isin(["ng"])
    ng_generation = black_generation.loc[condition_ng]

    condition_nuclear = generation['genfuel'].isin(["nuclear"])
    nuclear_generation = green_generation.loc[condition_nuclear]

    condition_hydro = generation['genfuel'].isin(["hydro"])
    hydro_generation = green_generation.loc[condition_hydro]

    condition_solar = generation['genfuel'].isin(["solar"])
    solar_generation = green_generation.loc[condition_solar]

    # map_wind_generators = np.zeros((len(wind_generation), NB))
    # bus_GN = np.array(wind_generation['bus'])
    # for i in range(len(wind_generation)):
    #     map_wind_generators[i, bus_GN[i]] = 1

    # map_hydro_generators = np.zeros((len(hydro_generation), NB))
    # bus_GN = np.array(hydro_generation['bus'])
    # for i in range(len(hydro_generation)):
    #     map_hydro_generators[i, bus_GN[i]] = 1

    # map_solar_generators = np.zeros((len(solar_generation), NB))
    # bus_GN = np.array(solar_generation['bus'])
    # for i in range(len(solar_generation)):
    #     map_solar_generators[i, bus_GN[i]] = 1

    # map_nuclear_generators = np.zeros((len(nuclear_generation), NB))
    # bus_GN = np.array(nuclear_generation['bus'])
    # for i in range(len(nuclear_generation)):
    #     map_nuclear_generators[i, bus_GN[i]] = 1

    # map_coal_generators = np.zeros((len(coal_generation), NB))
    # bus_GN = np.array(coal_generation['bus'])
    # for i in range(len(coal_generation)):
    #     map_coal_generators[i, bus_GN[i]] = 1

    # ng_generation = generation.loc[condition_ng]
    # map_ng_generators = np.zeros((len(ng_generation), NB))
    # bus_GN = np.array(ng_generation['bus'])
    # for i in range(len(ng_generation)):
    #     map_ng_generators[i, bus_GN[i]] = 1


    # Find the positions in the green_generators array for the wind-powered generators
    wind_indices = np.where(np.isin(green_generation, wind_generation))[0]
    map_wind_generators = np.zeros(len(green_generation))
    map_wind_generators[wind_indices] = 1

    # Find the indices in the green_generators array where the generator is hydro-powered
    hydro_indices = np.where(np.isin(green_generation, hydro_generation)[0])

    # Initialize the mapping array
    map_hydro_generators = np.zeros(len(green_generation))

    # Set the identified indices to 1 for hydro generators
    map_hydro_generators[hydro_indices] = 1

        # Find the indices in the green_generators array where the generator is solar-powered
    solar_indices = np.where(np.isin(green_generation, solar_generation))[0]

    # Initialize the mapping array
    map_solar_generators = np.zeros(len(green_generation))

    # Set the identified indices to 1 for solar generators
    map_solar_generators[solar_indices] = 1

    # Find the indices in the green_generators array where the generator is nuclear-powered
    nuclear_indices = np.where(np.isin(green_generation, nuclear_generation))[0]

    # Initialize the mapping array
    map_nuclear_generators = np.zeros(len(green_generation))

    # Set the identified indices to 1 for nuclear generators
    map_nuclear_generators[nuclear_indices] = 1

    # Find the indices in the green_generators array where the generator is coal-powered
    coal_indices = np.where(np.isin(black_generation, coal_generation))[0]

    # Initialize the mapping array
    map_coal_generators = np.zeros(len(black_generation))

    # Set the identified indices to 1 for coal generators
    map_coal_generators[coal_indices] = 1


        # Find the indices in the green_generators array where the generator is natural gas-powered
    ng_indices = np.where(np.isin(black_generation, ng_generation))[0]

    # Initialize the mapping array
    map_ng_generators = np.zeros(len(black_generation))

    # Set the identified indices to 1 for natural gas generators
    map_ng_generators[ng_indices] = 1

    # map_wind_generators = np.zeros((len(green_generation)))
    # # bus_GN = np.array(['bus'])
    # for i in range(len(wind_generation)):
    #     map_wind_generators[i] = 1

    # map_hydro_generators = np.zeros((len(hydro_generation), NB))
    # bus_GN = np.array(hydro_generation['bus'])
    # for i in range(len(hydro_generation)):
    #     map_hydro_generators[i, bus_GN[i]] = 1

    # map_solar_generators = np.zeros((len(solar_generation), NB))
    # bus_GN = np.array(solar_generation['bus'])
    # for i in range(len(solar_generation)):
    #     map_solar_generators[i, bus_GN[i]] = 1

    # map_nuclear_generators = np.zeros((len(nuclear_generation), NB))
    # bus_GN = np.array(nuclear_generation['bus'])
    # for i in range(len(nuclear_generation)):
    #     map_nuclear_generators[i, bus_GN[i]] = 1

    # map_coal_generators = np.zeros((len(coal_generation), NB))
    # bus_GN = np.array(coal_generation['bus'])
    # for i in range(len(coal_generation)):
    #     map_coal_generators[i, bus_GN[i]] = 1

    # ng_generation = generation.loc[condition_ng]
    # map_ng_generators = np.zeros((len(ng_generation), NB))
    # bus_GN = np.array(ng_generation['bus'])
    # for i in range(len(ng_generation)):
    #     map_ng_generators[i, bus_GN[i]] = 1

    map_green_generators = np.zeros((len(green_generation), NB))
    bus_GN = np.array(green_generation['bus'])
    for i in range(len(green_generation)):
        map_green_generators[i, bus_GN[i]] = 1

    map_black_generators = np.zeros((len(black_generation), NB))
    bus_GN = np.array(black_generation['bus'])
    for i in range(len(black_generation)):
        map_black_generators[i, bus_GN[i]] = 1

    max_green = np.array(green_generation['max_p_mw'].reset_index(drop=True))
    max_black = np.array(black_generation['max_p_mw'].reset_index(drop=True))
    
    return map_wind_generators, map_solar_generators, map_hydro_generators, map_coal_generators, map_nuclear_generators, map_ng_generators


def get_emissions(pg_green, pg_black, map_wind_generators, map_solar_generators, map_hydro_generators, map_coal_generators, map_nuclear_generators, map_ng_generators, baseMVA =100):
    # Emissions in kg/MWh
    emissions_coal = 855 
    emissions_ng = 690
    emissions_nuclear = 19
    emissions_hydro = 11
    emissions_solar = 101.5
    emissions_wind = 22

    emissions_green = pg_green*(map_wind_generators*emissions_wind+map_solar_generators*emissions_solar+map_nuclear_generators*emissions_nuclear+map_hydro_generators*emissions_hydro)
    emissions_black = pg_black*(map_coal_generators*emissions_coal+map_ng_generators*emissions_ng)

    emissions_green = sum(emissions_green)
    emissions_black = sum(emissions_black)

    total_emissions = emissions_green + emissions_black

    avg_emissions_per_MWh = total_emissions/((sum(pg_green)+sum(pg_black)))

    return avg_emissions_per_MWh, emissions_green, emissions_black