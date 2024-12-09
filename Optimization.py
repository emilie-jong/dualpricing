import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from Utilities_python import genfuel

def dc_opf(c, map_generators, map_loads, max_p_mw, min_p_mw, loads, len_loads,branch_cap, b, NG, NB, baseMVA, ref_bus=0):
    model = gp.Model()
    # scale_load = 0.5
    # pl = scale_load*loads.reset_index(drop=True)
    theta = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta{bus}") for bus in range(NB)]
    pg = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_p_mw[g], name=f"pg{g}") for g in range(NG)]
    pl = loads
    
    model.addConstr(theta[ref_bus]==0)
    model.update()
    
    obj = gp.quicksum(c[g]*pg[g] for g in range(NG)) *baseMVA
    model.setObjective(obj, gp.GRB.MINIMIZE)

    for n in range(NB):
        for m in range(NB):
            if (n != m):
                if (branch_cap[n][m]>0):
                    model.addConstr(b[n][m]*(theta[n]-theta[m]) == [-branch_cap[n][m], branch_cap[n][m]])


    #Definition of the constraints
    balance_constraint = {n:model.addConstr( # Balance equation expressed in p.u at each node
        -gp.quicksum(pg[g]*map_generators[g][n] for g in range((NG)))
        # -gp.quicksum(pg_green[g]*map_green_generators[g][n] for g in range(len(green_generation)))
        +gp.quicksum(b[n][m]*theta[m] for m in range(NB))
        +gp.quicksum(pl[g]*map_loads[g][n] for g in range(len_loads)),
        gp.GRB.EQUAL,
        0,name='Balance equation at node {0}'.format(n)) for n in range(NB)}




    # for n in range(NB):
    #     for m in range(NB):
    #         if (n != m):
    #             if (branch_cap[n][m]>0):
    #                 model.addConstr(b[n][m]*(theta[n]-theta[m]) == [-branch_cap[n][m], branch_cap[n][m]])
                    
    # #Definition of the constraints
    # balance_constraint = {n:model.addConstr( # Balance equation expressed in p.u at each node
    #     gp.quicksum(pg[g]*map_generators[g][n] for g in range(NG))
    #     -gp.quicksum(pl[g]*map_loads[g][n] for g in range(len_loads)),
    #     gp.GRB.EQUAL,
    #     gp.quicksum(b[n][m]*theta[m] for m in range(NB)),name='Balance equation at node {0}'.format(n)) for n in range(NB)}
    
    model.update()
    model.printStats()
    # model.display()
    model.optimize()

    return model, theta, pg, balance_constraint


def max_social_welfare_fixed_load_bid(load_bid, c, map_generators, map_loads, max_p_mw, min_p_mw, loads, len_loads, branch_cap, b, NG, NB, baseMVA, ref_bus=0):
    model = gp.Model()
    # scale_load = 0.5
    # pl = scale_load*loads.reset_index(drop=True)
    theta = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta{bus}") for bus in range(NB)]
    pg = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_p_mw[g], name=f"pg{g}") for g in range(NG)]
    pl = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    
    model.addConstr(theta[ref_bus]==0)
    model.update()
    
    obj = (gp.quicksum(load_bid*pl[g] for g in range(len_loads))-gp.quicksum(c[g]*pg[g] for g in range(NG))) *baseMVA
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    for n in range(NB):
        for m in range(NB):
            if (n != m):
                if (branch_cap[n][m]>0):
                    model.addConstr(b[n][m]*(theta[n]-theta[m]) == [-branch_cap[n][m], branch_cap[n][m]])
                    
    #Definition of the constraints
    balance_constraint = {n:model.addConstr( # Balance equation expressed in p.u at each node
        gp.quicksum(pg[g]*map_generators[g][n] for g in range(NG))
        -gp.quicksum(pl[g]*map_loads[g][n] for g in range(len_loads)),
        gp.GRB.EQUAL,
        gp.quicksum(b[n][m]*theta[m] for m in range(NB)),name='Balance equation at node {0}'.format(n)) for n in range(NB)}
    
    model.update()
    model.printStats()
    # model.display()
    model.optimize()

    return model, theta, pg, pl, balance_constraint

def opt_results(theta, pg, pl, balance_constraint, max_p_mw, NB, NG, len_loads):
    theta_ = []
    for i in range(NB):
        theta_.append(theta[i].X)
        # print(f'theta{i}:', theta[i].X)
    theta_ = np.array(theta_)

    pg_ = []
    for i in range(NG):
        # print(f'pg{i}:', pg[i].X)
        pg_.append(pg[i].X)
        # print(constraints[i].Pi)
    pg_=np.array(pg_)
    print('sum gen =', np.sum(pg_))
    print('sum load =', np.sum(pl[g].X for g in range(len_loads)))
        
    generation = pd.DataFrame()
    generation['p_mw'] = pg_
    generation['max_p_mw'] = max_p_mw
    generation['genfuel'] = genfuel
    condition_green = generation['genfuel'].isin(["wind", "hydro", "solar", "nuclear"])
    green_generation = generation.loc[condition_green]
    print('sum green gen =', np.sum(green_generation['p_mw']))
    print('max green gen =', np.sum(green_generation['max_p_mw']))
        
    condition_black = generation['genfuel'].isin(["coal", "ng"])
    black_generation = generation.loc[condition_black]
    print('sum black gen =', np.sum(black_generation['p_mw']))
    print('max black gen =', np.sum(black_generation['max_p_mw']))

    LMP_=[]
    for i in range(len(balance_constraint)):
        LMP_.append(-1*balance_constraint[i].Pi)
    LMPs = np.array(LMP_)
    return theta_, pg_, LMPs

def max_social_welfare_dual_dispatch( alpha, load_bid,green_generation, black_generation, max_green, max_black, map_black_generators, map_green_generators, map_loads, len_loads, loads, b, branch_cap, NB, NG, baseMVA, ref_bus=0):
    c_g = green_generation['c'].reset_index(drop=True)
    c_b = black_generation['c'].reset_index(drop=True)
    model = gp.Model()
    theta = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta{bus}") for bus in range(NB)]
    # pg = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_p_mw[g], name=f"pg{g}") for g in range(NG)]
    pg_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_green[g], name=f"pg{g}") for g in range(len(green_generation))]
    pg_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_black[g], name=f"pg{g}") for g in range(len(black_generation))]
    pl = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    # pl_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    # pl_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    model.addConstr(theta[ref_bus]==0)
    model.update()
    
    obj = (gp.quicksum(load_bid*pl[g] for g in range(len_loads))+gp.quicksum(alpha*pg_green[g] for g in range(len(green_generation)))-gp.quicksum(c_g[g]*pg_green[g] for g in range(len(green_generation)))- gp.quicksum(c_b[g]*pg_black[g] for g in range(len(black_generation)))) *baseMVA
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    
    # max_load_constraint = model.addConstr(gp.quicksum(pl_black[g]+pl_green[g])==loads[g] for g in range(len(loads)))
    # green_balance = model.addConstr(pg_green - pl_green == 0)
    
    for n in range(NB):
        for m in range(NB):
            if (n != m):
                if (branch_cap[n][m]>0):
                    model.addConstr(b[n][m]*(theta[n]-theta[m]) == [-branch_cap[n][m], branch_cap[n][m]])
                    
    #Definition of the constraints
    balance_constraint = {n:model.addConstr( # Balance equation expressed in p.u at each node
        gp.quicksum(pg_black[g]*map_black_generators[g][n] for g in range(len(black_generation)))
        +gp.quicksum(pg_green[g]*map_green_generators[g][n] for g in range(len(green_generation)))
        -gp.quicksum(pl[g]*map_loads[g][n] for g in range(len_loads)),
        gp.GRB.EQUAL,
        gp.quicksum(b[n][m]*theta[m] for m in range(NB)),name='Balance equation at node {0}'.format(n)) for n in range(NB)}
    
    model.update()
    model.printStats()
    # model.display()
    model.optimize()

    return model, theta, pg_green, pg_black, pl, balance_constraint

def opt_results_dual_dispatch(theta, pg_green, pg_black, pl, balance_constraint, green_generation, black_generation, len_loads, NB):
    theta_ = []
    for i in range(NB):
        theta_.append(theta[i].X)
        # print(f'theta{i}:', theta[i].X)
        
    pg_green_ = []
    
    for i in range(len(green_generation)):
        # print(f'pg{i}:', pg[i].X)
        pg_green_.append(pg_green[i].X)
        # print(constraints[i].Pi)
    pg_green_=np.array(pg_green_)
    
    pg_black_ = []
    for i in range(len(black_generation)):
        # print(f'pg{i}:', pg[i].X)
        pg_black_.append(pg_black[i].X)
        # print(constraints[i].Pi)
    pg_black_=np.array(pg_black_)
    print('sum black gen =', np.sum(pg_black_))
    print('sum green gen =', np.sum(pg_green_))
    print('sum load =', np.sum(pl[g].X for g in range(len_loads)))
        
    LMP_=[]
    for i in range(len(balance_constraint)):
        LMP_.append(-1*balance_constraint[i].Pi)
    
    return theta_, pg_black_, pg_green_, LMP_

def max_social_welfare_black_green_LMPs(alpha, load_bid,green_generation, black_generation, max_green, max_black, map_black_generators, map_green_generators, map_loads, len_loads, loads, b, branch_cap, NB, ref_bus=0):
    c_g = green_generation['c'].reset_index(drop=True)
    c_b = black_generation['c'].reset_index(drop=True)
    model = gp.Model()
    
    # scale_load = 0.5
    # pl = scale_load*loads.reset_index(drop=True)
    
    theta = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta{bus}") for bus in range(NB)]
    # pg = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_p_mw[g], name=f"pg{g}") for g in range(NG)]
    pg_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_green[g], name=f"pg{g}") for g in range(len(green_generation))]
    pg_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_black[g], name=f"pg{g}") for g in range(len(black_generation))]
    pl = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    pl_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    pl_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    diff_green = model.addVar()
    model.addConstr(theta[ref_bus]==0)
    model.update()
    
    obj = (gp.quicksum(load_bid[g]*pl[g] for g in range(len_loads))+gp.quicksum(alpha[g]*pl_green[g] for g in range(len_loads))-gp.quicksum(c_g[g]*pg_green[g] for g in range(len(green_generation)))- gp.quicksum(c_b[g]*pg_black[g] for g in range(len(black_generation))))
    model.setObjective(obj, gp.GRB.MAXIMIZE)

    for g in range(len_loads):
        max_load_constraint = model.addConstr(pl_black[g]+pl_green[g]==pl[g])

    green_balance = model.addConstr(gp.quicksum(pg_green[g] for g in range(len(green_generation))) - gp.quicksum(pl_green[g] for g in range(len_loads)) == 0)

    #     #Definition of the constraints
    # green_balance = {n:model.addConstr( # Balance equation expressed in p.u at each node
    #     gp.quicksum(pg_green[g]*map_green_generators[g][n] for g in range(len(green_generation)))
    #     -gp.quicksum(pl_green[g]*map_loads[g][n] for g in range(len_loads)),
    #     gp.GRB.EQUAL,
    #     gp.quicksum(b[n][m]*theta[m] for m in range(NB)),name='Balance equation at node {0}'.format(n)) for n in range(NB)}
    
    #         #Definition of the constraints
    # black_balance = {n:model.addConstr( # Balance equation expressed in p.u at each node
    #     gp.quicksum(pg_black[g]*map_black_generators[g][n] for g in range(len(black_generation)))
    #     -gp.quicksum(pl_black[g]*map_loads[g][n] for g in range(len_loads)),
    #     gp.GRB.EQUAL,
    #     gp.quicksum(b[n][m]*theta[m] for m in range(NB)),name='Balance equation at node {0}'.format(n)) for n in range(NB)}
    
    for n in range(NB):
        for m in range(NB):
            if (n != m):
                if (branch_cap[n][m]>0):
                    model.addConstr(b[n][m]*(theta[n]-theta[m]) == [-branch_cap[n][m], branch_cap[n][m]])


    #Definition of the constraints
    balance_constraint = {n:model.addConstr( # Balance equation expressed in p.u at each node
        gp.quicksum(pg_black[g]*map_black_generators[g][n] for g in range(len(black_generation)))
        +gp.quicksum(pg_green[g]*map_green_generators[g][n] for g in range(len(green_generation)))
        -gp.quicksum(pl[g]*map_loads[g][n] for g in range(len_loads)),
        gp.GRB.EQUAL,
        gp.quicksum(b[n][m]*theta[m] for m in range(NB)),name='Balance equation at node {0}'.format(n)) for n in range(NB)}


    model.update()
    model.printStats()
    # model.display()
    model.optimize()    

    return model, theta, pg_green, pg_black, pl_green, pl_black, balance_constraint, green_balance

def max_social_welfare_black_green_LMPs_v3(alpha, load_bid,green_generation, black_generation, max_green, max_black, map_black_generators, map_green_generators, map_loads, len_loads, loads, b, branch_cap, NB, ref_bus=0):
    c_g = green_generation['c'].reset_index(drop=True)
    c_b = black_generation['c'].reset_index(drop=True)
    model = gp.Model()
    
    # scale_load = 0.5
    # pl = scale_load*loads.reset_index(drop=True)
    
    theta = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta{bus}") for bus in range(NB)]
    # pg = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_p_mw[g], name=f"pg{g}") for g in range(NG)]
    pg_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_green[g], name=f"pg_g{g}") for g in range(len(green_generation))]
    pg_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_black[g], name=f"pg_b{g}") for g in range(len(black_generation))]
    pl = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    pl_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl_b{g}") for g in range(len_loads)]
    pl_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl_g{g}") for g in range(len_loads)]
    model.addConstr(theta[ref_bus]==0)
    model.update()
    
    obj = (gp.quicksum(load_bid[g]*pl[g] for g in range(len_loads))+gp.quicksum(alpha[g]*pl_green[g] for g in range(len_loads))-gp.quicksum(c_g[g]*pg_green[g] for g in range(len(green_generation)))- gp.quicksum(c_b[g]*pg_black[g] for g in range(len(black_generation))))
    model.setObjective(obj, gp.GRB.MAXIMIZE)

    max_load_constraint = {g:model.addConstr(pl_black[g]+pl_green[g]==pl[g]) for g in range(len_loads)}

    green_balance = model.addConstr(gp.quicksum(pl_green[g] for g in range(len_loads))-gp.quicksum(pg_green[g] for g in range(len(green_generation))), gp.GRB.EQUAL, 0)
    #black_balance = model.addConstr( gp.quicksum(pl_black[g] for g in range(len_loads))-gp.quicksum(pg_black[g] for g in range(len(black_generation))), gp.GRB.EQUAL, 0)

    for n in range(NB):
        for m in range(NB):
            if (n != m):
                if (branch_cap[n][m]>0):
                    model.addConstr(b[n][m]*(theta[n]-theta[m]) == [-branch_cap[n][m], branch_cap[n][m]])


    #Definition of the constraints
    balance_constraint = {n:model.addConstr( # Balance equation expressed in p.u at each node
        -gp.quicksum(pg_black[g]*map_black_generators[g][n] for g in range(len(black_generation)))
        -gp.quicksum(pg_green[g]*map_green_generators[g][n] for g in range(len(green_generation)))
        +gp.quicksum(b[n][m]*(theta[n]-theta[m]) for m in range(NB))
        +gp.quicksum(pl[g]*map_loads[g][n] for g in range(len_loads)),
        gp.GRB.EQUAL,
        0,name='Balance equation at node {0}'.format(n)) for n in range(NB)}


    model.update()
    model.printStats()
    # model.display()
    model.optimize()    

    return model, theta, pg_green, pg_black, pl_green, pl_black, balance_constraint, green_balance

def max_social_welfare_black_green_LMPs_v4(alpha, load_bid,green_generation, black_generation, max_green, max_black, map_black_generators, map_green_generators, map_loads, len_loads, loads, b, branch_cap, NB, ref_bus=0):
    c_g = green_generation['c'].reset_index(drop=True)
    c_b = black_generation['c'].reset_index(drop=True)
    model = gp.Model()
    
    # scale_load = 0.5
    # pl = scale_load*loads.reset_index(drop=True)
    
    theta = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta{bus}") for bus in range(NB)]
    # pg = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_p_mw[g], name=f"pg{g}") for g in range(NG)]
    pg_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_green[g], name=f"pg_g{g}") for g in range(len(green_generation))]
    pg_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_black[g], name=f"pg_b{g}") for g in range(len(black_generation))]
    pl = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    pl_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl_b{g}") for g in range(len_loads)]
    pl_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl_g{g}") for g in range(len_loads)]
    model.addConstr(theta[ref_bus]==0)
    model.update()
    
    obj = (gp.quicksum(load_bid[g]*pl[g] for g in range(len_loads))+gp.quicksum(alpha[g]*pl_green[g] for g in range(len_loads))-gp.quicksum(c_g[g]*pg_green[g] for g in range(len(green_generation)))- gp.quicksum(c_b[g]*pg_black[g] for g in range(len(black_generation))))
    model.setObjective(obj, gp.GRB.MAXIMIZE)

    max_load_constraint = {g:model.addConstr(pl_black[g]+pl_green[g]==pl[g]) for g in range(len_loads)}

    green_balance = model.addConstr(gp.quicksum(pl_green[g] for g in range(len_loads))-gp.quicksum(pg_green[g] for g in range(len(green_generation))), gp.GRB.EQUAL, 0)
    black_balance = model.addConstr( gp.quicksum(pl_black[g] for g in range(len_loads))-gp.quicksum(pg_black[g] for g in range(len(black_generation))), gp.GRB.EQUAL, 0)

    for n in range(NB):
        for m in range(NB):
            if (n != m):
                if (branch_cap[n][m]>0):
                    model.addConstr(b[n][m]*(theta[n]-theta[m]) == [-branch_cap[n][m], branch_cap[n][m]])


    #Definition of the constraints
    balance_constraint_green = {n:model.addConstr( # Balance equation expressed in p.u at each node
        -gp.quicksum(pg_green[g]*map_green_generators[g][n] for g in range(len(green_generation)))
        +gp.quicksum(b[n][m]*theta[m] for m in range(NB))
        +gp.quicksum(pl_green[g]*map_loads[g][n] for g in range(len_loads)),
        gp.GRB.EQUAL,
        0,name='Balance equation at node {0}'.format(n)) for n in range(NB)}
    
    balance_constraint_black = {n:model.addConstr( # Balance equation expressed in p.u at each node
        -gp.quicksum(pg_black[g]*map_black_generators[g][n] for g in range(len(black_generation)))
        -gp.quicksum(pg_green[g]*map_green_generators[g][n] for g in range(len(green_generation)))
        +gp.quicksum(b[n][m]*theta[m] for m in range(NB))
        +gp.quicksum(pl[g]*map_loads[g][n] for g in range(len_loads)),
        gp.GRB.EQUAL,
        0,name='Balance equation at node {0}'.format(n)) for n in range(NB)}


    model.update()
    model.printStats()
    # model.display()
    model.optimize()    

    return model, theta, pg_green, pg_black, pl_green, pl_black, balance_constraint_green, green_balance, black_balance, balance_constraint_black

def opt_results_black_green_LMPs(theta, pg_green, pg_black, pl_green, pl_black, balance_constraint, green_balance, green_generation, black_generation, len_loads, NB):
    theta_ = []
    for i in range(NB):
        theta_.append(theta[i].X)
        # print(f'theta{i}:', theta[i].X)
        
    pg_green_ = []
    
    for i in range(len(green_generation)):
        # print(f'pg{i}:', pg[i].X)
        pg_green_.append(pg_green[i].X)
        # print(constraints[i].Pi)
    pg_green_=np.array(pg_green_)
    
    pg_black_ = []
    for i in range(len(black_generation)):
        # print(f'pg{i}:', pg[i].X)
        pg_black_.append(pg_black[i].X)
        # print(constraints[i].Pi)
    pg_black_=np.array(pg_black_)
    print('sum black gen =', np.sum(pg_black_))
    print('sum green gen =', np.sum(pg_green_))
    
    pl_green_ = []
    pl_black_ = []
    for i in range(len_loads):
        pl_green_.append(pl_green[i].X)
        pl_black_.append(pl_black[i].X)
        # print(constraints[i].Pi)
    pl_green_=np.array(pg_green_)
    pl_black_ = np.array(pl_black_)
    print('sum black load =', np.sum(pl_black_))
    print('sum green load =', np.sum(pl_green_))
        
    LMP_=[]
    for i in range(len(balance_constraint)):
        LMP_.append(-1*balance_constraint[i].Pi)
    
    # lambda_green =[]
    # for i in range(len(balance_constraint)):
    #     lambda_green.append(-1*green_balance[i].Pi)



    lambda_green = -1*green_balance.Pi
    
    return theta_, pg_black_, pg_green_, pl_green_, pl_black_, LMP_, lambda_green

def opt_results_black_green_LMPs_v3(theta, pg_green, pg_black, pl_green, pl_black, balance_constraint, green_balance, green_generation, black_generation, len_loads, NB):
    theta_ = []
    for i in range(NB):
        theta_.append(theta[i].X)
        # print(f'theta{i}:', theta[i].X)
        
    pg_green_ = []
    
    for i in range(len(green_generation)):
        # print(f'pg{i}:', pg[i].X)
        pg_green_.append(pg_green[i].X)
        # print(constraints[i].Pi)
    pg_green_=np.array(pg_green_)
    
    pg_black_ = []
    for i in range(len(black_generation)):
        # print(f'pg{i}:', pg[i].X)
        pg_black_.append(pg_black[i].X)
        # print(constraints[i].Pi)
    pg_black_=np.array(pg_black_)
    print('sum black gen =', np.sum(pg_black_))
    print('sum green gen =', np.sum(pg_green_))
    
    from decimal import Decimal, getcontext

    # Set the precision high enough to capture very small values
    getcontext().prec = 50

    pl_green_ = []
    pl_black_ = []
    for i in range(len_loads):
        pl_green_.append(pl_green[i].X)
        pl_black_.append(pl_black[i].X)
        # print(constraints[i].Pi)
    pl_green_=np.array(pg_green_, dtype=np.float64)
    pl_black_ = np.array(pl_black_, dtype=np.float64)
    pl_black_decimal = [Decimal(x) for x in pl_black_]
    sum_pl_black = sum(pl_black_decimal)
    #np.sum(pl_black_)
    print('sum black load =', sum_pl_black)
    print('sum green load =', np.sum(pl_green_))
        
    LMP_=[]
    for i in range(len(balance_constraint)):
        LMP_.append(balance_constraint[i].Pi)
    
    # lambda_green =[]
    # for i in range(len(balance_constraint)):
    #     lambda_green.append(-1*green_balance[i].Pi)



    lambda_green = green_balance.Pi
    # lambda_black = black_balance.Pi
    
    return theta_, pg_black_, pg_green_, pl_green_, pl_black_, LMP_, lambda_green


## Different green LMPs
def max_social_welfare_black_green_LMPs2(alpha, load_bid,green_generation, black_generation, max_green, max_black, map_black_generators, map_green_generators, map_loads, len_loads, loads, b, branch_cap, NB, ref_bus=0):
    c_g = green_generation['c'].reset_index(drop=True)
    c_b = black_generation['c'].reset_index(drop=True)
    model = gp.Model()
    
    # scale_load = 0.5
    # pl = scale_load*loads.reset_index(drop=True)
    
    theta = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta{bus}") for bus in range(NB)]
    # pg = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_p_mw[g], name=f"pg{g}") for g in range(NG)]
    pg_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_green[g], name=f"pg{g}") for g in range(len(green_generation))]
    pg_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_black[g], name=f"pg{g}") for g in range(len(black_generation))]
    pl = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    pl_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    pl_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    diff_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"diff_green{g}") for g in range(NB)]
    diff_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"diff_black{g}") for g in range(NB)]
    model.addConstr(theta[ref_bus]==0)
    model.update()
    
    obj = (gp.quicksum(load_bid[g]*pl[g] for g in range(len_loads))+gp.quicksum(alpha[g]*pl_green[g] for g in range(len_loads))-gp.quicksum(c_g[g]*pg_green[g] for g in range(len(green_generation)))- gp.quicksum(c_b[g]*pg_black[g] for g in range(len(black_generation))))
    model.setObjective(obj, gp.GRB.MAXIMIZE)

    for g in range(len_loads):
        max_load_constraint = model.addConstr(pl_black[g]+pl_green[g]==pl[g])

    green_balance = model.addConstr(gp.quicksum(pg_green[g] for g in range(len(green_generation))) - gp.quicksum(pl_green[g] for g in range(len_loads)) == 0)

        #     #Definition of the constraints
    green_difference = {n:model.addConstr( # Balance equation expressed in p.u at each node
        gp.quicksum(pg_green[g]*map_green_generators[g][n] for g in range(len(green_generation)))
        -gp.quicksum(pl_green[g]*map_loads[g][n] for g in range(len_loads)),
        gp.GRB.EQUAL,
        diff_green[n],name='Balance equation at node {0}'.format(n)) for n in range(NB)}
    
            #     #Definition of the constraints
    black_difference = {n:model.addConstr( # Balance equation expressed in p.u at each node
        gp.quicksum(pg_black[g]*map_black_generators[g][n] for g in range(len(black_generation)))
        -gp.quicksum(pl_black[g]*map_loads[g][n] for g in range(len_loads)),
        gp.GRB.EQUAL,
        diff_black[n],name='Balance equation at node {0}'.format(n)) for n in range(NB)}
    
    difference = {n:model.addConstr( # Balance equation expressed in p.u at each node
        diff_green[n]+diff_black[n],
        gp.GRB.EQUAL,
        gp.quicksum(b[n][m]*theta[m] for m in range(NB)),name='Difference equation at node {0}'.format(n)) for n in range(NB)}

    
    for n in range(NB):
        for m in range(NB):
            if (n != m):
                if (branch_cap[n][m]>0):
                    model.addConstr(b[n][m]*(theta[n]-theta[m]) == [-branch_cap[n][m], branch_cap[n][m]])


    # #Definition of the constraints
    # balance_constraint = {n:model.addConstr( # Balance equation expressed in p.u at each node
    #     gp.quicksum(pg_black[g]*map_black_generators[g][n] for g in range(len(black_generation)))
    #     +gp.quicksum(pg_green[g]*map_green_generators[g][n] for g in range(len(green_generation)))
    #     -gp.quicksum(pl[g]*map_loads[g][n] for g in range(len_loads)),
    #     gp.GRB.EQUAL,
    #     gp.quicksum(b[n][m]*theta[m] for m in range(NB)),name='Balance equation at node {0}'.format(n)) for n in range(NB)}


    model.update()
    model.printStats()
    # model.display()
    model.optimize()    

    return model, theta, pg_green, pg_black, pl_green, pl_black, green_balance, green_difference, black_difference, difference

def opt_results_black_green_LMPs2(theta, pg_green, pg_black, pl_green, pl_black, green_balance, green_difference, black_difference, difference, green_generation, black_generation, len_loads, NB):
    theta_ = []
    for i in range(NB):
        theta_.append(theta[i].X)
        # print(f'theta{i}:', theta[i].X)
        
    pg_green_ = []
    
    for i in range(len(green_generation)):
        # print(f'pg{i}:', pg[i].X)
        pg_green_.append(pg_green[i].X)
        # print(constraints[i].Pi)
    pg_green_=np.array(pg_green_)
    
    pg_black_ = []
    for i in range(len(black_generation)):
        # print(f'pg{i}:', pg[i].X)
        pg_black_.append(pg_black[i].X)
        # print(constraints[i].Pi)
    pg_black_=np.array(pg_black_)
    print('sum black gen =', np.sum(pg_black_))
    print('sum green gen =', np.sum(pg_green_))
    
    pl_green_ = []
    pl_black_ = []
    for i in range(len_loads):
        pl_green_.append(pl_green[i].X)
        pl_black_.append(pl_black[i].X)
        # print(constraints[i].Pi)
    pl_green_=np.array(pg_green_)
    pl_black_ = np.array(pl_black_)
    print('sum black load =', np.sum(pl_black_))
    print('sum green load =', np.sum(pl_green_))
        
    # LMP_=[]
    # for i in range(len(balance_constraint)):
    #     LMP_.append(-1*balance_constraint[i].Pi)
    
    # lambda_green =[]
    # for i in range(len(balance_constraint)):
    #     lambda_green.append(-1*green_balance[i].Pi)
    
    difference_ = []
    green_difference_ = []
    black_difference_ = []
    for i in range(len(difference)):
        difference_.append(-1*difference[i].Pi)
        green_difference_.append(-1*green_difference[i].Pi)
        black_difference_.append(-1*black_difference[i].Pi)

    diff = pd.DataFrame()
    # diff['LMP'] = LMP_
    diff['difference'] = difference_
    diff['green_diff'] = green_difference_
    diff['black_diff'] = black_difference_
    diff['green_plus_black'] = [x+y for x, y in zip(green_difference_,black_difference_)]

    lambda_green = -1*green_balance.Pi
    
    return theta_, pg_black_, pg_green_, pl_green_, pl_black_, lambda_green, diff

def results_dc_opf(theta, pg, balance_constraint, NB):
    theta_ = []
    for i in range(NB):
        theta_.append(theta[i].X)
    
    pg_ = []
    for i in range(len(pg)):
        pg_.append(pg[i].X)
        # print(f'theta{i}:', theta[i].X)

    LMP_=[]
    for i in range(len(balance_constraint)):
           LMP_.append(balance_constraint[i].Pi)


    return theta_, pg_, LMP_


def max_social_welfare_black_green_LMPs_v5(alpha, load_bid,green_generation, black_generation, max_green, max_black, map_black_generators, map_green_generators, map_loads, len_loads, loads, b, branch_cap, NB, ref_bus=0):
    c_g = green_generation['c'].reset_index(drop=True)
    c_b = black_generation['c'].reset_index(drop=True)
    model = gp.Model()
    
    # scale_load = 0.5
    # pl = scale_load*loads.reset_index(drop=True)
    
    theta = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta{bus}") for bus in range(NB)]
    # pg = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_p_mw[g], name=f"pg{g}") for g in range(NG)]
    pg_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_green[g], name=f"pg_g{g}") for g in range(len(green_generation))]
    pg_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_black[g], name=f"pg_b{g}") for g in range(len(black_generation))]
    pl = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl{g}") for g in range(len_loads)]
    pl_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl_b{g}") for g in range(len_loads)]
    pl_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=loads[g], name=f"pl_g{g}") for g in range(len_loads)]
    epsilon_green = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub = GRB.INFINITY, name=f"epsilon_g{g}") for g in range(NB)]
    epsilon_black = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub = GRB.INFINITY, name=f"epsilon_b{g}") for g in range(NB)]
    model.addConstr(theta[ref_bus]==0)
    model.update()
    
    obj = (gp.quicksum(load_bid[g]*pl[g] for g in range(len_loads))+gp.quicksum(alpha[g]*pl_green[g] for g in range(len_loads))-gp.quicksum(c_g[g]*pg_green[g] for g in range(len(green_generation)))- gp.quicksum(c_b[g]*pg_black[g] for g in range(len(black_generation))))
    model.setObjective(obj, gp.GRB.MAXIMIZE)

    max_load_constraint = {g:model.addConstr(pl_black[g]+pl_green[g]==pl[g]) for g in range(len_loads)}

    green_balance = model.addConstr(gp.quicksum(pl_green[g] for g in range(len_loads))-gp.quicksum(pg_green[g] for g in range(len(green_generation))), gp.GRB.EQUAL, 0)
    black_balance = model.addConstr( gp.quicksum(pl_black[g] for g in range(len_loads))-gp.quicksum(pg_black[g] for g in range(len(black_generation))), gp.GRB.EQUAL, 0)

    for n in range(NB):
        for m in range(NB):
            if (n != m):
                if (branch_cap[n][m]>0):
                    model.addConstr(b[n][m]*(theta[n]-theta[m]) == [-branch_cap[n][m], branch_cap[n][m]])

    # model.addConstr(epsilon_green + epsilon_black == [-branch_cap[n][m], branch_cap[n][m]] for m in range (NB))
    # model.addConstr(epsilon_green[n] + epsilon_black[n] == gp.quicksum(branch_cap[n][m] for m in range(NB)))
    epsilon_balance = {n:model.addConstr(epsilon_green[n] + epsilon_black[n] == gp.quicksum(b[n][m]*(theta[n]-theta[m]) for m in range(NB))) for n in range(NB)}
    epsilon_min_green = {n:model.addConstr(epsilon_green[n] >= -1*gp.quicksum((b[n][m]*(theta[n]-theta[m]) for m in range(NB))))}
    epsilon_max_green = {n:model.addConstr(epsilon_green[n] <= gp.quicksum((b[n][m]*(theta[n]-theta[m]) for m in range(NB))))}
    epsilon_min_black = {n:model.addConstr(epsilon_black[n] >= -1*gp.quicksum((b[n][m]*(theta[n]-theta[m]) for m in range(NB))))}
    epsilon_max_black = {n:model.addConstr(epsilon_black[n] <= gp.quicksum((b[n][m]*(theta[n]-theta[m]) for m in range(NB))))}

    # epsilon_max_green = {n:model.addConstr(epsilon_green[n] == [-1*gp.quicksum(b[n][m]*(theta[n]-theta[m]) for m in range(NB)), gp.quicksum(b[n][m]*(theta[n]-theta[m]) for m in range(NB))] for n in range(NB))}
    # epsilon_max_black = {n:model.addConstr(epsilon_black[n] == [-1*gp.quicksum(b[n][m]*(theta[n]-theta[m]) for m in range(NB)), gp.quicksum(b[n][m]*(theta[n]-theta[m]) for m in range(NB))] for n in range(NB))}
    # model.addConstr([-branch_cap[n][m], branch_cap[n][m]])


    #Definition of the constraints
    green_balance_constraint = {n:model.addConstr( # Balance equation expressed in p.u at each node
        -gp.quicksum(pg_green[g]*map_green_generators[g][n] for g in range(len(green_generation)))
        +gp.quicksum(b[n][m]*(theta[n]-theta[m]) for m in range(NB))
        -epsilon_green[n]
        +gp.quicksum(pl_green[g]*map_loads[g][n] for g in range(len_loads)),
        gp.GRB.EQUAL,
        0,name='Balance equation at node {0}'.format(n)) for n in range(NB)}

    black_balance_constraint = {n:model.addConstr( # Balance equation expressed in p.u at each node
        -gp.quicksum(pg_black[g]*map_black_generators[g][n] for g in range(len(black_generation)))
        +gp.quicksum(b[n][m]*(theta[n]-theta[m]) for m in range(NB))
        -epsilon_black[n]
        +gp.quicksum(pl_black[g]*map_loads[g][n] for g in range(len_loads)),
        gp.GRB.EQUAL,
        0,name='Balance equation at node {0}'.format(n)) for n in range(NB)}


    model.update()
    model.printStats()
    # model.display()
    model.optimize()    

    return model, theta, pg_green, pg_black, pl_green, pl_black, black_balance_constraint, green_balance_constraint, epsilon_balance, epsilon_green, epsilon_black, black_balance

def opt_results_black_green_LMPs_v5(theta, pg_green, pg_black, pl_green, pl_black, black_balance_constraint, green_balance_constraint, epsilon_balance, epsilon_green, epsilon_black, green_generation, black_generation, len_loads, NB):
    theta_ = []
    for i in range(NB):
        theta_.append(theta[i].X)
        # print(f'theta{i}:', theta[i].X)
        
    pg_green_ = []
    
    for i in range(len(green_generation)):
        # print(f'pg{i}:', pg[i].X)
        pg_green_.append(pg_green[i].X)
        # print(constraints[i].Pi)
    pg_green_=np.array(pg_green_)
    
    pg_black_ = []
    for i in range(len(black_generation)):
        # print(f'pg{i}:', pg[i].X)
        pg_black_.append(pg_black[i].X)
        # print(constraints[i].Pi)
    pg_black_=np.array(pg_black_)
    print('sum black gen =', np.sum(pg_black_))
    print('sum green gen =', np.sum(pg_green_))
    
    pl_green_ = []
    pl_black_ = []
    for i in range(len_loads):
        pl_green_.append(pl_green[i].X)
        pl_black_.append(pl_black[i].X)
        # print(constraints[i].Pi)
    pl_green_=np.array(pg_green_)
    pl_black_ = np.array(pl_black_)
    print('sum black load =', np.sum(pl_black_))
    print('sum green load =', np.sum(pl_green_))

    # green_LMP_=[]
    # for i in range(len(green_balance_constraint)):
    #     green_LMP_.append(green_balance_constraint[i].Pi)
    
    # black_LMP_ = []
    # for i in range(len(black_balance_constraint)):
    #     black_LMP_.append(black_balance_constraint[i].Pi)

        
    # LMP_=[]
    # for i in range(len(balance_constraint)):
    #     LMP_.append(balance_constraint[i].Pi)
    
    # lambda_green =[]
    # for i in range(len(balance_constraint)):
    #     lambda_green.append(-1*green_balance[i].Pi)



    # lambda_green = green_balance.Pi
    # lambda_black = black_balance.Pi

    lambda_epsilon =[]
    for i in range(len(epsilon_balance)):
        lambda_epsilon.append(-1*epsilon_balance[i].Pi)
    
    lambda_epsilon = np.array(lambda_epsilon)

    black_LMP_ = []
    for i in range(len(epsilon_balance)):
        # print(f'pg{i}:', pg[i].X)
        black_LMP_.append(lambda_epsilon[i]*(epsilon_green[i].X)/(epsilon_green[i].X + epsilon_black[i].X))
    
    green_LMP_ = []
    for i in range(len(epsilon_balance)):
        # print(f'pg{i}:', pg[i].X)
        green_LMP_.append(lambda_epsilon[i]*(epsilon_black[i].X)/(epsilon_green[i].X + epsilon_black[i].X))
    
    return theta_, pg_black_, pg_green_, pl_green_, pl_black_, black_LMP_, green_LMP_, lambda_epsilon