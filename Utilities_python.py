import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog




def solve_SCED(Pload, DAM_model):
    # Call model parameters
    Pgen_green_min = DAM_model['Pgen_green_min']
    Pgen_green_max = DAM_model['Pgen_green_max']
    Pgen_black_min = DAM_model['Pgen_black_min']
    Pgen_black_max = DAM_model['Pgen_black_max']
    Pgen_green_bid = DAM_model['Pgen_green_bid']
    Pgen_black_bid = DAM_model['Pgen_black_bid']
    alpha = np.array(DAM_model['alpha'])
    flow_lim = DAM_model['flow_lim']
    flow_lim_max = np.array(DAM_model['flow_lim'])
    flow_lim_min = -np.array(DAM_model['flow_lim'])
    PTDF = np.array(DAM_model['PTDF'])
    num_line = PTDF.shape[0]
    num_bus = PTDF.shape[1]

    model = gp.Model()

    Pgen_green = model.addVars(range(num_bus), lb=Pgen_green_min, ub=Pgen_green_max, name="Pgen_green")
    Pgen_black = model.addVars(range(num_bus), lb=Pgen_black_min, ub=Pgen_black_max, name="Pgen_black")
    Pload_green = model.addVars(range(num_bus), lb=0.0, ub=Pload, name="Pload_green")

    # Balance equations
    model.addConstr(sum(Pload_green[i] for i in range(num_bus)) - sum(Pgen_green[i] for i in range(num_bus)) == 0, "Balance_green")
    model.addConstr(sum(Pload[i] for i in range(num_bus)) - sum(Pgen_green[i] for i in range(num_bus)) - sum(Pgen_black[i] for i in range(num_bus)) == 0, "Balance_black")

    # Flow limits
    flow_expression = [gp.LinExpr() for _ in range(num_line)]
    for i in range(num_line):
        for j in range(num_bus):
            flow_expression[i] += PTDF[i][j] * (Pgen_black[j] + Pgen_green[j] - Pload[j])
        model.addRange(flow_expression[i],flow_lim_min[i],flow_lim_max[i], f"Flow_limit_{i}")
    
        # Objective function
    obj = gp.LinExpr()
    for i in range(num_bus):
        obj.add(Pgen_black[i] * Pgen_black_bid[i] + Pgen_green[i] * Pgen_green_bid[i] - alpha[i] * Pload_green[i])
    model.setObjective(obj, GRB.MINIMIZE)

    # Solve the problem
    model.optimize()

    SCED = {
        'Pgen_green': [Pgen_green[i].x for i in range(num_bus)],
        'Pgen_black': [Pgen_black[i].x for i in range(num_bus)],
        'Pload_green': [Pload_green[i].x for i in range(num_bus)]
    }

    return SCED

def compute_LMPs(primal, DAM_model):
    # Take a primal solution and compute the associated duals
    # with associated minimized LMPs.
    #
    # For now, assume single bids and single loads+gens/bus

    Pload = primal['Pload']
    Pgen = primal['Pgen']

    # Call model parameters
    Pgen_min = DAM_model['Pgen_min']
    Pgen_max = DAM_model['Pgen_max']
    Pgen_bid = DAM_model['Pgen_bid']
    flow_lim = DAM_model['flow_lim']
    flow_lim_max = np.array(DAM_model['flow_lim'])
    flow_lim_min = -np.array(DAM_model['flow_lim'])
    PTDF = np.matrix(DAM_model['PTDF'])
    num_line = PTDF.shape[0]
    num_bus = PTDF.shape[1]


    model = gp.Model()


    mu_min_gen = model.addMVar(num_bus, lb = 0, name = "mu_min_gen")
    mu_max_gen = model.addMVar(num_bus, lb = 0, name = "mu_max_gen")
    mu_min_flow = model.addMVar(num_line, lb = 0, name = "mu_min_flow")
    mu_max_flow = model.addMVar(num_line, lb = 0, name = "mu_max_flow")

    # mu_min_gen = model.addVars(range(num_bus), lb = 0, name = "mu_min_gen")
    # mu_max_gen = model.addVars(range(num_bus), lb = 0, name = "mu_max_gen")
    # mu_min_flow = model.addVars(range(num_line), lb = 0, name = "mu_min_flow")
    # mu_max_flow = model.addVars(range(num_line), lb = 0, name = "mu_max_flow")

    lambda_ = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="lambda_")
    # model.update()

    # # # Now, loop and apply assign duals to 0 if their constraint isn't active!
    for bus in range(num_bus):
        # Black output -- lower
        if abs(Pgen[bus] - Pgen_min[bus]) < 1e-6:
            # Constraint active!
            pass
        else:
            model.addConstr(mu_min_gen[bus] == 0)

        # Black output -- upper
        if abs(Pgen[bus] - Pgen_max[bus]) < 1e-6:
            # Constraint active!
            pass
        else:
           model.addConstr(mu_max_gen[bus] == 0)
    
    model.update()


    # # Loop over lines  
    for line in range(num_line):
        # Flow limit -- lower
        if abs(flow_lim_min[line] - PTDF[line, :].T*(Pgen - Pload)).any() < 1e-6:
            # Constraint active!
            pass
        else:
            model.addConstr(mu_min_flow[line] == 0)

        # Flow limit -- upper
        if abs(PTDF[line, :].T*(Pgen - Pload) - flow_lim_max[line]).any() < 1e-6:
            # Constraint active!
            pass
        else:
            model.addConstr(mu_max_flow[line] == 0)
    
    model.update()

    # Stationarity -- Loop and take the derivative with respect to all
    # primals in the RT market clearing problem
    for bus in range(num_bus):
        # The following assumes the PTDF has been padded with zeros at the
        # the location (column) of the reference bus.
        model.addConstr(Pgen_bid[bus] - mu_min_gen[bus] + mu_max_gen[bus] +- lambda_ - 
             mu_min_flow.transpose()*PTDF[:, bus] + mu_max_flow.transpose()*PTDF[:, bus] ==0)
        model.addConstr(Pgen_bid[bus] - mu_min_gen[bus] + mu_max_gen[bus] +- lambda_ - 
             mu_min_flow.transpose()*PTDF[:, bus] + mu_max_flow.transpose()*PTDF[:, bus] ==0)

    model.update()

    # Now, compute the LMPs
    all_LMP = lambda_ + (mu_min_flow.transpose() - mu_max_flow.transpose())*PTDF
    
    # Now, minimize the LMPs (l1 norm)
    obj = 0
    for bus in range(num_bus):
        LMP = lambda_+(mu_min_flow.transpose()-mu_max_flow.transpose())*PTDF[:,bus]
        tmp = model.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, ub=GRB.INFINITY)
        model.addConstr(+LMP <= tmp)
        model.addConstr(-LMP <= tmp)
        # obj.add(tmp)
        obj+= tmp

    model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    model.optimize()

    if model.SolCount < 1:
        print("No feasible solution found")

    model.display()

    # model.computeIIS()
    # print('\nThe following constraints and variables are in the IIS:')
    # for c in model.getConstrs():
    #     if c.IISConstr: print(f'\t{c.constrname}: {model.getRow(c)} {c.Sense} {c.RHS}')

    # for v in model.getVars():
    #     if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
    #     if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')



    # Return the full model
    LMP = {
        'LMP': all_LMP.getValue(),
        #'LMP': all_LMP.x,
        #'LMP': [mu.x for mu in all_LMP],
        'lambda': lambda_.x,
        'mu_min_flow': mu_min_flow.getAttr(GRB.Attr.X),
        'mu_max_flow': mu_max_flow.getAttr(GRB.Attr.X),
        'mu_min_gen': mu_min_gen.getAttr(GRB.Attr.X),
        'mu_max_gen': mu_max_gen.getAttr(GRB.Attr.X),

    }
    


    # Output
    return LMP


def compute_dual_LMPs(primal, DAM_model):
    # Take a primal solution and compute the associated duals
    # with associated minimized LMPs.
    #
    # For now, assume single bids and single loads+gens/bus

    Pload = primal['Pload']
    Pload_green = primal['Pload_green']
    Pgen_green  = primal['Pgen_green']
    Pgen_black  = primal['Pgen_black']

    # Call model parameters
    Pgen_green_min = DAM_model['Pgen_green_min']
    Pgen_green_max = DAM_model['Pgen_green_max']
    Pgen_black_min = DAM_model['Pgen_black_min']
    Pgen_black_max = DAM_model['Pgen_black_max']
    Pgen_green_bid = DAM_model['Pgen_green_bid']
    Pgen_black_bid = DAM_model['Pgen_black_bid']
    alpha = DAM_model['alpha']
    flow_lim = DAM_model['flow_lim']
    flow_lim_max = np.array(DAM_model['flow_lim'])
    flow_lim_min = -np.array(DAM_model['flow_lim'])
    PTDF = np.array(DAM_model['PTDF'])
    num_line = PTDF.shape[0]
    num_bus = PTDF.shape[1]

    model = gp.Model()
    mu_min_green = model.addMVar(num_bus, lb = 0, name = "mu_min_green")
    mu_max_green = model.addMVar(num_bus, lb = 0, name = "mu_max_green")
    mu_min_black = model.addMVar(num_bus, lb = 0, name = "mu_min_black")
    mu_max_black = model.addMVar(num_bus, lb = 0, name = "mu_max_black")
    mu_min_flow = model.addMVar(num_line, lb = 0, name = "mu_min_flow")
    mu_max_flow = model.addMVar(num_line, lb = 0, name = "mu_max_flow")
    mu_min_a = model.addMVar(num_bus, lb = 0, name = "mu_min_a")
    mu_max_a = model.addMVar(num_bus, lb = 0, name = "mu_max_a")

    lambda_green = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="lambda_green")
    lambda_black = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="lambda_black")
    # model.update()

    # # # Now, loop and apply assign duals to 0 if their constraint isn't active!
    for bus in range(num_bus):
        # Black output -- lower
        if abs(Pgen_green[bus] - Pgen_green_min[bus]) < 1e-6:
            # Constraint active!
            pass
        else:
            model.addConstr(mu_min_green[bus] == 0)

        # Black output -- upper
        if abs(Pgen_green[bus] - Pgen_green_max[bus]) < 1e-6:
            # Constraint active!
            pass
        else:
           model.addConstr(mu_max_green[bus] == 0)
    
    model.update()

    for bus in range(num_bus):
        # Black output -- lower
        if abs(Pgen_black[bus] - Pgen_black_min[bus]) < 1e-6:
            # Constraint active!
            pass
        else:
            model.addConstr(mu_min_black[bus] == 0)

        # Black output -- upper
        if abs(Pgen_black[bus] - Pgen_black_max[bus]) < 1e-6:
            # Constraint active!
            pass
        else:
           model.addConstr(mu_max_black[bus] == 0)
    
    model.update()

    for bus in range(num_bus):
        # Black output -- lower
        if abs(Pload_green[bus]) < 1e-6:
            # Constraint active!
            pass
        else:
            model.addConstr(mu_min_a[bus] == 0)

        # Black output -- upper
        if abs(Pload_green[bus] - Pload[bus]) < 1e-6:
            # Constraint active!
            pass
        else:
           model.addConstr(mu_max_a[bus] == 0)
    
    model.update()


    # Loop over lines
    for line in range(num_line):
        # Flow limit -- lower
        if abs(flow_lim_min[line] - PTDF[line, :].T*(Pgen_black+Pgen_green - Pload)).any() < 1e-6:
            # Constraint active!
            pass
        else:
            model.addConstr(mu_min_flow[line] == 0)

        # Flow limit -- upper
        if abs(PTDF[line, :].T*(Pgen_black+Pgen_green - Pload) - flow_lim_max[line]).any() < 1e-6:
            # Constraint active!
            pass
        else:
            model.addConstr(mu_max_flow[line] == 0)
    
    model.update()

    # Stationarity -- Loop and take the derivative with respect to all
    # primals in the RT market clearing problem
    # (1) Pload_green (green allocation), (2) Pgen_green, (3) Pgen_black
    for bus in range(num_bus):
        if bus != 4:
           # (1) Pload_green (green allocation)
            model.addConstr(-alpha[bus] + mu_max_a[bus] - mu_min_a[bus] + lambda_green == 0.0)
            # The following assumes the PTDF has been padded with zeros at the
            # the location (column) of the reference bus.
         # (2) Pgen_green
            model.addConstr(Pgen_green_bid[bus] - mu_min_green[bus] + mu_max_green[bus] - lambda_green - lambda_black 
                            - mu_min_flow.transpose()*PTDF[:, bus] + mu_max_flow.transpose()*PTDF[:, bus] ==0)
         # (3) Pgen_green
            model.addConstr(Pgen_black_bid[bus] - mu_min_black[bus] + mu_max_black[bus] - lambda_black - 
                            mu_min_flow.transpose()*PTDF[:, bus] + mu_max_flow.transpose()*PTDF[:, bus] ==0)

    model.update()

    # Now, compute the LMPs
    all_green_LMP = lambda_green + lambda_black+ (mu_min_flow.transpose() - mu_max_flow.transpose())*PTDF
    all_black_LMP = lambda_black+ (mu_min_flow.transpose() - mu_max_flow.transpose())*PTDF

    # Now, minimize the LMPs (l1 norm)
    obj = 0

    for bus in range(num_bus):
        green_LMP = lambda_green + lambda_black+(mu_min_flow.transpose()-mu_max_flow.transpose())*PTDF[:,bus]
        black_LMP = lambda_black+(mu_min_flow.transpose()-mu_max_flow.transpose())*PTDF[:,bus]
        tmp = model.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, ub=GRB.INFINITY)
        model.addConstr(+green_LMP <= tmp)
        model.addConstr(-green_LMP <= tmp)
        obj+= tmp
        model.addConstr(+black_LMP <= tmp)
        model.addConstr(-black_LMP <= tmp)
        obj+= tmp
    model.setObjective(obj, sense=gp.GRB.MINIMIZE)
    model.optimize()

    # model.computeIIS()
    # print('\nThe following constraints and variables are in the IIS:')
    # for c in model.getConstrs():
    #     if c.IISConstr: print(f'\t{c.constrname}: {model.getRow(c)} {c.Sense} {c.RHS}')

    # for v in model.getVars():
    #     if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
    #     if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')


    # Return the full model
    Dual_LMP = {
        'green_LMP': green_LMP.getValue(),
        'black_LMP': black_LMP.getValue(),
        'lambda_green': lambda_green.x,
        'lambda_black': lambda_black.x
    }

    # Output
    return Dual_LMP


## Copy of genfuel matrix in m file

genfuel = [
	'wind',
	'wind',
	'wind',
	'solar',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'wind',
	'ng',
	'ng',
	'solar',
	'solar',
	'wind',
	'wind',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'ng',
	'ng',
	'ng',
	'wind',
	'wind',
	'ng',
	'ng',
	'wind',
	'coal',
	'coal',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'ng',
	'ng',
	'ng',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'ng',
	'ng',
	'ng',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'hydro',
	'hydro',
	'wind',
	'wind',
	'wind',
	'wind',
	'wind',
	'solar',
	'wind',
	'wind',
	'wind',
	'wind',
	'ng',
	'ng',
	'ng',
	'wind',
	'wind',
	'wind',
	'coal',
	'coal',
	'hydro',
	'hydro',
	'hydro',
	'wind',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'coal',
	'coal',
	'coal',
	'coal',
	'coal',
	'coal',
	'coal',
	'coal',
	'coal',
	'wind',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'wind',
	'wind',
	'ng',
	'ng',
	'ng',
	'ng',
	'wind',
	'wind',
	'wind',
	'wind',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'wind',
	'wind',
	'wind',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'hydro',
	'coal',
	'coal',
	'ng',
	'ng',
	'wind',
	'wind',
	'ng',
	'ng',
	'ng',
	'wind',
	'wind',
	'nuclear',
	'nuclear',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'hydro',
	'hydro',
	'coal',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'wind',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'hydro',
	'hydro',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'coal',
	'coal',
	'coal',
	'hydro',
	'hydro',
	'solar',
	'solar',
	'solar',
	'solar',
	'solar',
	'solar',
	'ng',
	'ng',
	'ng',
	'coal',
	'coal',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'hydro',
	'hydro',
	'hydro',
	'hydro',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'hydro',
	'hydro',
	'hydro',
	'hydro',
	'hydro',
	'hydro',
	'hydro',
	'hydro',
	'hydro',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'solar',
	'solar',
	'solar',
	'solar',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'nuclear',
	'nuclear',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'solar',
	'solar',
	'solar',
	'solar',
	'solar',
	'solar',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'coal',
	'coal',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'solar',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'ng',
	'coal',
	'coal',
	'coal',
	'coal',
	'ng',
	'ng',
	'ng',
	'ng',
	'coal',
	'coal',
	'solar',
	'coal',
	'coal',
	'coal',
	'coal',
	'coal',
	'coal',
	'coal',
	'coal',
	'ng',
	'coal',
	'coal',
	'ng',
    'ng']