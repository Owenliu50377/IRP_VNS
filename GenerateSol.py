
import re
import math
from operator import itemgetter
import random
import numpy as np
## STAGE 1 for generating init sol

## input parameters
N = [0,1,2,3]# ALL the nodes
RC = list(range(1,21)) # retail center // 0 is reserved for the depot, and 1,2... for RC
H = (len(RC)+1) * [0.5] # cost of inventory per unit of each RC_i
S = (len(RC)+1) * [2] # cost of stock out per unit of each RC_i

T = [0,1,2,3,4,5] # Time horizon start from 1 // 0,1,2,3,4...
C = (len(RC)+1) * [200] # Max inventory capacity for each RC_i
Q = [] # matrix with |RC| * |T| the random demand of RC_i at time_t 
Policy = [0.25, 0.5, 0.75, 1] # Refill policies for C_i e.g 25% 50% 75%
INIT_r = []

for m in range(len(Policy)):
    b= [] # reserved for depot
    for i in range(len(RC)):
        a = []
        for j in range(len(T)):
            a.append(Policy[m])
        b.append(a)
    INIT_r.append(b)

I = [["depot"]] # 2-dimensional list , the inventory level of RC_i at time t

d = [["depot"]] # 2-dimensional list , the supplyment of RC_I at time t d[0][t] = 0 depot
for i in range(len(RC)):
    I.append([50])
    d.append([])
Risk = {} # risk level for different level of refill policies


# Feas_interval =[ [[1,10],[2,9],],[] ] # Target interval for inventory of RC_i at time t
#                                         # 0 stands for the lower bound and 1 for the upper bound
# Feas_interval = []


V = 40000 # capacity of each car
ci0 = 0 # cost of empty car back up

f = 400 # fixed cost of using a car associated with V
##ASSUMING only one car to use
Total_shippment_cost = 0 # costt of transportation


## data loading...
location = open("location.txt",mode = "r")
rows = list(list(map(float, re.split(' ', row))) for row in re.split('\n', location.read())[:-1])

#loading distance between each node
distance = []
cost_ship = [] # cost of shippment distance
for i in range(len(rows)):
    a = []
    m = []
    for j in range(len(rows)):
        if i == j:
            a.append(0)
        else:
            b =  math.sqrt( (rows[i][0] - rows[j][0])**2 + (rows[i][1] - rows[j][1])**2)
            a.append(b)
            m.append(b/100)
    distance.append(a)
    cost_ship.append(m)


demand = open("Demand.txt",mode = "r")
Q = list(list(map(float, re.split(' ', row))) for row in re.split('\n', demand.read())[:-1])
Q.insert(0,["depot"]) ## depot elimination
  
## 
## r = matrix of the supplyment 
## Phase 1 
## Calculate the risk of violation of the inventory level of every Refill Policy
def Phase1_2(r):

    global Risk
    global Total_shippment_cost
    global d
    risk = 0
    Risk[str(r)] = 0
    cost_inventory = 0
    cost_stockout = 0
    for i in RC:
        # total risk
        accumrisk_ir = 0
        # cost of violation
        inventory = 0 
        stockout = 0

        for t in T:
            # update supplyment level

            up_d = max(r[i-1][t] * C[i] - I[i][t], 0)
            d[i].append(up_d)

            # update Inventory level at next time period
            up_i = max(I[i][t] + up_d - Q[i][t], 0)
            ## WARNING:
            ## Qit has a bigg difference with I
            ##
            ##
            ##
            ####
            # always 0
            I[i].append(up_i)


            risk_itr = 0
            coeff_risk = 0.2
            Max_Inven = 200
            if (up_i < Max_Inven * coeff_risk) or (up_i > Max_Inven):
                risk_itr = min(np.abs(up_i - Max_Inven), np.abs(Max_Inven * coeff_risk - up_i)) 

            #
            inventory += max( I[i][0]\
                 + sum(d[i] \
                [0: t+1]) - sum(Q[i][0: t+1]), 0)
            stockout += max(-I[i][0] - sum(d[i][0: t+1]) + sum(Q[i][0: t+1]), 0)
            accumrisk_ir += risk_itr

            
        ## WAITING 
        ## codes here for calculating cost of inventory and Stockout cost
        cost_inventory += H[i] * inventory
        cost_stockout += S[i] * stockout
        # Update Risk[r]
        Risk[str(r)] += accumrisk_ir

    risk = Risk[str(r)]
    ## Phase 2 C-W Saving Algorithm
    ## CW algorithm
    for t in T:

        saving = 0
        routes = [] # Routes 
        savings = [] # savings total
        for i in range(1, len(RC)+1):
            routes.append([i])
            # 1,2,3,4,5 RC number...

        for i in range(1, len(routes) + 1):
            for j in range(1, len(routes) + 1):
                if i != j:
                    saving = distance[i][0] + distance[0][j] - distance[i][j]
                    savings.append([i,j,saving])

        ## SAVINGS 2-d matrix for node i and node j.

        savings = sorted(savings, key= itemgetter(2), reverse = True)

        for i in range(len(savings)):
            startRoute = []
            endRoute = []
            routeDemand = 0
            for j in range(len(routes)):

                if (savings[i][1] == routes[j][0]):
                    startRoute = routes[j]
                elif savings[i][0] == routes[j][-1]:
                    endRoute = routes[j]

                if ((len(startRoute) != 0)) and (len(endRoute) != 0):
                    for k in range(len(startRoute)):
                        routeDemand += d[startRoute[k]][t]
                    for k in range(len(endRoute)):
                        routeDemand += d[endRoute[k]][t]
                    routeDistance = 0
                    routestore = [0]+endRoute+startRoute+[0]
                    for i in range(len(routestore)-1):
                    
                        routeDistance += distance[routestore[i]][routestore[i+1]]

                    if (routeDemand <= V):   
                        routes.remove(startRoute)
                        routes.remove(endRoute)
                        routes.append(endRoute + startRoute)

                    break

             # Cost of transportation


        for i in range(len(routes)):
            routes[i].insert(0, 0)
            routes[i].insert(len(routes[i]), 0)

        ## routes construction
        for i in routes:
            dist = 0
            for j in range(len(i) - 1):
                dist += distance[i[j]][i[j+1]]

            # print("routes", i, ' distance', dist)


        # print(d)
        xit = 0 # number of pass arc 
        for i in range(len(routes)):
            route_demand = 0
            xit += len(routes[i]) - 1
            # n routes
            for j in routes[i]:
                if j == 0: continue # depot no demand
                else:
                    route_demand += d[j][t]

            # print(route_demand)
            for j in range(len(routes[i]) - 1):
                # j = start node
                # cost of the intermediate transportation
                Total_shippment_cost +=  route_demand * cost_ship[routes[i][j]][routes[i][j+1]]
                # print(d)
                if route_demand > 0:
                   
                    route_demand -= d[routes[i][j+1]][t]
                else:
                    # empty cost
                    Total_shippment_cost += ci0 * 1
            ##fixed cost of a single car

        Total_shippment_cost += f * xit

        TOTAL_COST = Total_shippment_cost + cost_inventory + cost_stockout

    return TOTAL_COST, risk


def Phase3(B = 1e100):
    base_sol = None
    elite_sol = []
    for i in range(len(INIT_r)):
        policy = INIT_r[i]
        totalcost = Phase1_2(policy)[0]
        if totalcost > B :
            continue
        else:
            elite_sol.append(policy)
    

    best_value = float("inf")
    for i in range(len(elite_sol)):
        cost, risk = Phase1_2(elite_sol[i])
        
        if cost < best_value:
            base_sol = elite_sol[i]
            best_value = cost


    return base_sol, elite_sol



# base_sol, elite_sol = Phase3()

# M = 100 # maximum iterations of VNS
# eliteSize = 5
# k_max = len(RC) * len(T)# maximum percentages of policies to reset ?? |S| * |T|
# # elite_sol = [base_sol]
# maxRuns = 10

def shaking(k , sol):
    newSol = sol
    # generate random value to substitute the value of matrix
    for i in range(k):
        cons = random.random()
        row = random.randint(0, len(RC) - 1)
        col = random.randint(0, len(T) - 1)
        newSol[row][col] = cons


    return newSol


def local_search(sol):
    current_cost, current_risk = Phase1_2(sol)
    
    new_risk = float("inf")
    
    m = 1
    M = 100
    newSol = sol
    row = random.randint(0, len(RC) - 1)
    col = random.randint(0, len(T) - 1)
    
    while new_risk > current_risk and m < M:
        a = random.random()
        newSol[row][col] = a
        new_cost, new_risk = Phase1_2(newSol)
        m += 1

    return newSol

def worstSol(eliteSol):
    new_risk = float('-inf')
    worstSol = None
    for i in eliteSol:
        current_cost, risk = Phase1_2(i)
        if risk > new_risk:
            worstSol = i
            new_risk = risk

    return worstSol
        

def VNS():
    base_sol, elite_sol = Phase3()
    print(Phase1_2(base_sol))
    M = 100 # maximum iterations of VNS
    eliteSize = 5
    k_max = len(RC) * len(T)# maximum percentages of policies to reset ?? |S| * |T|
    # elite_sol = [base_sol]
    maxRuns = 10
    ## VNS

    k = 1
    iter = 0
    while k < k_max and iter <= maxRuns:
        
        newSol = shaking(k,base_sol)
        #print("FINISH SHAKING")
        newSol = local_search(newSol)
        #print("Start local search")
        if len(elite_sol) < eliteSize:
            #print("ADD to elite solution due to empty set")
            elite_sol.append(newSol)
            iter += 1


        elif Phase1_2(newSol)[1] < Phase1_2(worstSol(elite_sol))[1]:
            #print("ADD to elite solution due to substitution")
            elite_sol.remove(worstSol(elite_sol))
            elite_sol.append(newSol)
            iter += 1


        if Phase1_2(newSol)[1] < Phase1_2(base_sol)[1]:
            base_sol = newSol
            k = 1
        else:
            k += 1  

    ## Refinement of best solution 
    best_sol = base_sol

    for i in elite_sol:
        if Risk[str(i)] < Risk[str(best_sol)]:
            best_sol = i

            
    return best_sol


print(VNS())
print(Phase1_2(VNS()))
print(1)
