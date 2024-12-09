import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from Utilities_python import genfuel



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

