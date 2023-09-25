import re
import math
from operator import itemgetter
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook
import xlsxwriter
import copy
from pylab import mpl
mpl.rc("font",family='Songti SC')
#mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

## STAGE 1 for generating init sol
'##------------------------------input parameters------------------------------##'
## input parameters
# N = [0,1,2,3]# ALL the nodes
RC = list(range(1, 21))  # retail center // 0 is reserved for the depot, and 1,2... for RC
H = (len(RC) + 1) * [0.06]  # cost of inventory per unit of each RC_i
S = (len(RC) + 1) * [0.15]  # cost of stock out per unit of each RC_i

T = [0, 1, 2, 3, 4, 5]  # Time horizon start from 1 // 0,1,2,3,4...
C = (len(RC) + 1) * [100000]  # Max inventory capacity for each RC_i
Q = []  # matrix with |RC| * |T| the random demand of RC_i at time_t
Policy = [0.25, 0.5, 0.75, 1]  # Refill policies for C_i e.g 25% 50% 75%
INIT_r = []

for m in range(len(Policy)):
    b = []  # reserved for depot
    for i in range(len(RC)):
        a = []
        for j in range(len(T)):
            a.append(Policy[m])
        b.append(a)
    INIT_r.append(b)

# I = [["depot"]]  # 2-dimensional list , the inventory level of RC_i at time t
# d = [["depot"]]  # 2-dimensional list , the supplyment of RC_I at time t d[0][t] = 0 depot
# for i in range(len(RC)):
#     I.append([20000])
#     d.append([])
Risk = {}  # risk level for different level of refill policies
InvCost = {}
StoCost = {}
RouCost = {}
TotalCost = {}

# Feas_interval =[ [[1,10],[2,9],],[] ] # Target interval for inventory of RC_i at time t
#                                         # 0 stands for the lower bound and 1 for the upper bound
# Feas_interval = []


V = 50000  # capacity of each car
ci0 = 100  # cost of empty car back up

f = 200  # fixed cost of using a car associated with V
##ASSUMING only one car to use
Total_shippment_cost = 0  # costt of transportation

## data loading...
location = open("/Users/liushuting/Desktop/研究生/毕业论文/算法/数据/location_N109.txt", mode="r")
rows = list(list(map(float, re.split(' ', row))) for row in re.split('\n', location.read())[:-1])

# loading distance between each node
distance = []
cost_ship = []  # cost of shippment distance
for i in range(len(rows)):
    a = []
    m = []
    for j in range(len(rows)):
        if i == j:
            a.append(0)
        else:
            b = math.sqrt((rows[i][0] - rows[j][0]) ** 2 + (rows[i][1] - rows[j][1]) ** 2)
            a.append(b)
            m.append(b / 10000)
    distance.append(a)
    cost_ship.append(m)

demand = open("/Users/liushuting/Desktop/研究生/毕业论文/算法/数据/Demand_N109.txt", mode="r")
Q = list(list(map(float, re.split(' ', row))) for row in re.split('\n', demand.read())[:-1])
Q.insert(0, ["depot"])  ## depot elimination

B = 200000

# 记录风险变化和成本变化
list_totalcost = []
list_invcost = []
list_stocost = []
list_roucost = []
list_risk = []
list_RiskLevel = []
list_Times = []
list_CoverLevel = []

'##-------------------------------Phase1-2-----------------------------##'
## r = matrix of the supplyment 
## Phase 1 
## Calculate the risk of violation of the inventory level of every Refill Policy
def Phase1_2(r):
    global Risk
    global d
    global InvCost
    global StoCost
    global RouCost
    global TotalCost
    risk = 0
    invcost = 0
    stocost = 0
    totalshipcost = 0
    totalcost = 0
    Risk[str(r)] = 0
    InvCost[str(r)] = 0
    StoCost[str(r)] = 0
    RouCost[str(r)] = 0
    TotalCost[str(r)] = 0

    I = [["depot"]]  # 2-dimensional list , the inventory level of RC_i at time t
    d = [["depot"]]  # 2-dimensional list , the supplyment of RC_I at time t d[0][t] = 0 depot
    for i in range(len(RC)):
        I.append([20000])
        d.append([])
    times = 0
    for i in RC:
        # total risk
        accumrisk_ir = 0
        # cost of violation
        accuminvcost = 0
        accumstocost = 0

        for t in T:
            # update supplyment level
            up_d = max(r[i - 1][t] * C[i] - I[i][t], 0)
            d[i].append(up_d)
            # update Inventory level at next time period
            up_i = max(I[i][t] + up_d - Q[i][t], 0)
            I[i].append(up_i)

            risk_itr = 0
            coeff_risk = 0.2
            Max_Inven = 100000
            if (up_i < Max_Inven * coeff_risk) or (up_i > Max_Inven):
                risk_itr = min(np.abs(up_i - Max_Inven), np.abs(Max_Inven * coeff_risk - up_i))
                times += 1

            # 计算库存成本和缺货成本
            inventory = up_i
            stockout = max(-I[i][t] - up_d + Q[i][t], 0)
            accumrisk_ir += risk_itr
            accuminvcost += H[i] * inventory
            accumstocost += S[i] * stockout

        InvCost[str(r)] += accuminvcost
        StoCost[str(r)] += accumstocost
        # Update Risk[r]
        Risk[str(r)] += accumrisk_ir

    # 处理list
    risk = Risk[str(r)]
    # list_risk.append(risk)
    # # 加入list_risk
    # if list_risk == []:
    #     list_risk.append(risk)
    # else:
    #     if risk <= min(list_risk):
    #         list_risk.append(risk)

    invcost = InvCost[str(r)]
    # list_invcost.append(invcost)
    # # 加入list_invcost
    # if list_invcost == []:
    #     list_invcost.append(invcost)
    # else:
    #     if invcost <= min(list_invcost):
    #         list_invcost.append(invcost)

    stocost = StoCost[str(r)]
    # list_stocost.append(stocost)
    # # 加入list_stocost
    # if list_stocost == []:
    #     list_stocost.append(stocost)
    # else:
    #     if stocost <= min(list_stocost):
    #         list_stocost.append(stocost)


    '##-------------------------------Phase 2-----------------------------##'
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

        for i in range(len(routes)):
            routes[i].insert(0, 0)
            routes[i].insert(len(routes[i]), 0)
        ## routes construction
        for i in routes:
            dist = 0
            for j in range(len(i) - 1):
                dist += distance[i[j]][i[j+1]]
            # print("routes", i, ' distance', dist)

        '##-------------------------------得到路径后计算结果-----------------------------##'
        # print(d)
        xit = 0  # number of pass arc 计算固定成本所用
        back_cost = 0  # 一个周期内的空车返回成本
        ship_cost = 0  # 一个周期内的总路径成本
        for i in range(len(routes)):
            xit += len(routes[i]) - 1
            back_cost += ci0 * cost_ship[routes[i][-2]][routes[i][-1]]
            route_demand = 0
            roucost = 0  # 一条路径的成本
            # one_route_demand = []
            # n routes
            for j in routes[i]:
                if j == 0:
                    continue  # depot no demand
                else:
                    route_demand += d[j][t]
                    # one_route_demand.append(d[j][t])

            for j in range(len(routes[i]) - 1):
                roucost += route_demand * cost_ship[routes[i][j]][routes[i][j + 1]]
                m = routes[i][j + 1]
                if route_demand > 0 and m != 0:
                    try:
                        route_demand -= d[m][t]
                    except IndexError or TypeError or KeyError:
                        print(routes)
                        print(m)
                        # print(d)

            # print(route_demand)
            # print(roucost)
            ship_cost += roucost
        RouCost[str(r)] += f * xit + back_cost + ship_cost  # 所有周期所有运输成本

    totalshipcost = RouCost[str(r)]
    # list_roucost.append(totalshipcost)
    # # 加入list_roucost
    # if list_roucost == []:
    #     list_roucost.append(totalshipcost)
    # else:
    #     if Total_shippment_cost <= min(list_roucost):
    #         list_roucost.append(totalshipcost)

    TotalCost[str(r)] = totalshipcost + invcost + stocost
    totalcost = TotalCost[str(r)]
    # list_totalcost.append(totalcost)
    # # 加入list_totalcost
    # if list_totalcost == []:
    #     list_totalcost.append(totalcost)
    # else:
    #     if totalcost <= min(list_totalcost):
    #         list_totalcost.append(totalcost)
    risklevel = risk / matrixsum(Q)
    coverlevel = times / (len(RC) * len(T))

    return totalcost, risk, times, risklevel, coverlevel, I, d


'##-------------------------------Phase 3-----------------------------##'
def Phase3():
    base_sol = None
    elite_sol = []
    # B = 200000
    for i in range(len(INIT_r)):
        policy = INIT_r[i]
        totalcost = Phase1_2(policy)[0]
        if totalcost <= B:
            elite_sol.append(policy)

    best_value = float("inf")
    for i in range(len(elite_sol)):
        cost, risk, _, _, _, _, _ = Phase1_2(elite_sol[i])
        if cost < best_value:
            base_sol = elite_sol[i]
            best_value = cost
    return base_sol, elite_sol



'##-------------------------------shaking-----------------------------##'
# M = 100 # maximum iterations of VNS
# eliteSize = 5
# k_max = len(RC) * len(T)# maximum percentages of policies to reset ?? |S| * |T|
# # elite_sol = [base_sol]
# maxRuns = 10
def shaking(k , sol):
    newSol = sol
    index = []
    i = 1
    # generate random value to substitute the value of matrix
    while i <= k:
        row = random.randint(0, len(RC) - 1)
        col = random.randint(0, len(T) - 1)
        if (row,col) in index:
            continue
        else:
            i+=1
    
    for i in index:
        cons = random.random()
        newSol[i[0]][i[1]] = cons
        
### ??????? 确保替换三个不同的值

    return newSol

def repeat(M,sol):
    new_risk = float("inf")
    m = 1
    newSol = copy.deepcopy(sol)
    row = random.randint(0, len(RC) - 1)
    col = random.randint(0, len(T) - 1)
    bestsol = None
    # B = 200000
    while m < M:
        a = random.random()
        # row = random.randint(0, len(RC) - 1)
        # col = random.randint(0, len(T) - 1)
        newSol[row][col] = a
        current_cost, current_risk, _, _, _, _, _ = Phase1_2(newSol)
        if current_risk < new_risk and current_cost <= B:
            bestsol = newSol
            new_cost, new_risk, _, _, _, _, _ = Phase1_2(bestsol)
        m += 1
    return bestsol

'##-------------------------------local_search-----------------------------##'
def local_search(sol):
    # current_cost, current_risk, _ = Phase1_2(sol)
    n = 100
    while n <= 300:
        bestsol = repeat(n, sol)
        if bestsol == None:
            n += 50
            bestsol = repeat(n,sol)
        else:
            break

    return bestsol



'##-------------------------------worstSol-----------------------------##'
def worstSol(eliteSol):
    new_risk = float('-inf')
    worstSol = None
    for i in eliteSol:
        current_cost, risk, _, _, _, _, _ = Phase1_2(i)
        if risk > new_risk:
            worstSol = i
            new_risk = risk

    return worstSol
        


'##-------------------------------VNS-----------------------------##'
def VNS():
    base_sol, elite_sol = Phase3()

    print(Phase1_2(base_sol)[0:5])
    list_totalcost.append(Phase1_2(base_sol)[0])
    list_risk.append(Phase1_2(base_sol)[1])
    list_Times.append(Phase1_2(base_sol)[2])
    list_RiskLevel.append(Phase1_2(base_sol)[3])
    list_CoverLevel.append(Phase1_2(base_sol)[4])

    eliteSize = 5
    k_max = len(RC) * len(T)# maximum percentages of policies to reset ?? |S| * |T|
    # elite_sol = [base_sol]
    maxRuns = 100
    k = 1
    iter = 0
    # B = 200000
    while k < k_max and iter <= maxRuns:
        
        newSol = shaking(k,base_sol)
        #print("FINISH SHAKING")
        newSol = local_search(newSol)
        cost_newsol = Phase1_2(newSol)[0]

        #print("Start local search")
        if len(elite_sol) < eliteSize  and cost_newsol <= B:
            #print("ADD to elite solution due to empty set")
            elite_sol.append(newSol)
            list_totalcost.append(Phase1_2(base_sol)[0])
            list_risk.append(Phase1_2(base_sol)[1])
            list_Times.append(Phase1_2(base_sol)[2])
            list_RiskLevel.append(Phase1_2(base_sol)[3])
            list_CoverLevel.append(Phase1_2(base_sol)[4])
            iter += 1


        elif Phase1_2(newSol)[1] < Phase1_2(worstSol(elite_sol))[1] and cost_newsol <= B:
            #print("ADD to elite solution due to substitution")
            elite_sol.remove(worstSol(elite_sol))
            elite_sol.append(newSol)
            list_totalcost.append(Phase1_2(base_sol)[0])
            list_risk.append(Phase1_2(base_sol)[1])
            list_Times.append(Phase1_2(base_sol)[2])
            list_RiskLevel.append(Phase1_2(base_sol)[3])
            list_CoverLevel.append(Phase1_2(base_sol)[4])

            # # 加入list_risk
            # if list_risk == []:
            #     list_risk.append(Phase1_2(newSol)[1])
            # else:
            #     if Phase1_2(newSol)[1] < min(list_risk):
            #         list_risk.append(Phase1_2(newSol)[1])

            iter += 1


        if Phase1_2(newSol)[1] < Phase1_2(base_sol)[1] and cost_newsol <= B:
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


'##-------------------------------调用函数、计算指标-----------------------------------##'
# print(VNS())
def matrixsum(a):
    sum = 0
    for i in range(len(a)):
        for j in range(len(a[i])):
            if type(a[i][j]) == float:
                sum += int(a[i][j])
    return sum

BestSol = VNS()
SumCost, TotalRisk, Times, RiskLevel, CoverLevel, I_output, d_output = Phase1_2(BestSol)
# RiskLevel = TotalRisk / matrixsum(Q)
# CoverLevel = Times / (len(RC) * len(T))


'##-------------------------------绘制折线图-----------------------------------##'
x_totalcost = range(len(list_totalcost))
y_totolcost = list_totalcost
x_risk = range(len(list_risk))
y_risk = list_risk
x_times = range(len(list_Times))
y_times = list_Times
x_risklevel = range(len(list_RiskLevel))
y_risklevel = list_RiskLevel
x_coverlevel = range(len(list_CoverLevel))
y_coverlevel = list_CoverLevel
x_invcost = range(len(list_invcost))
y_invcost = list_invcost
x_stocost = range(len(list_stocost))
y_stocost = list_stocost
x_roucost = range(len(list_roucost))
y_roucost = list_roucost


fig = plt.figure(figsize =(10,6))
fig.add_subplot(3,2,1)
plt.plot(x_totalcost, # x轴数据
         y_totolcost, # y轴数据
         linestyle = '-', # 折线类型
         linewidth = 0.5, # 折线宽度
         color = 'steelblue', # 折线颜色
         marker = 'o', # 折线图中添加圆点
         markersize = 3, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='brown', # 点的填充色
         label = 'totalcost'
         )
plt.title("总成本变化图")
fig.add_subplot(3,2,2)
plt.plot(x_risk, # x轴数据
         y_risk, # y轴数据
         linestyle = '-', # 折线类型
         linewidth = 0.5, # 折线宽度
         color = 'steelblue', # 折线颜色
         marker = 'o', # 折线图中添加圆点
         markersize = 3, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='brown', # 点的填充色
         label = 'risk'
         )
plt.title("风险变化图")
fig.add_subplot(3,2,3)
plt.plot(x_times, # x轴数据
         y_times, # y轴数据
         linestyle = '-', # 折线类型
         linewidth = 0.5, # 折线宽度
         color = 'steelblue', # 折线颜色
         marker = 'o', # 折线图中添加圆点
         markersize = 3, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='brown', # 点的填充色
         label = 'times'
         )
plt.title("违背要求次数变化图")
# fig.add_subplot(3,2,4)
# plt.plot(x_risklevel, # x轴数据
#          y_risklevel, # y轴数据
#          linestyle = '-', # 折线类型
#          linewidth = 0.5, # 折线宽度
#          color = 'steelblue', # 折线颜色
#          marker = 'o', # 折线图中添加圆点
#          markersize = 3, # 点的大小
#          markeredgecolor='black', # 点的边框色
#          markerfacecolor='brown', # 点的填充色
#          label = 'risklevel'
#          )
# fig.add_subplot(3,2,5)
# plt.plot(x_coverlevel, # x轴数据
#          y_coverlevel, # y轴数据
#          linestyle = '-', # 折线类型
#          linewidth = 0.5, # 折线宽度
#          color = 'steelblue', # 折线颜色
#          marker = 'o', # 折线图中添加圆点
#          markersize = 3, # 点的大小
#          markeredgecolor='black', # 点的边框色
#          markerfacecolor='brown', # 点的填充色
#          label = 'coverlevel'
#          )
# fig.add_subplot(3,2,6)
# plt.plot(x_invcost, # x轴数据
#          y_invcost, # y轴数据
#          linestyle = '-', # 折线类型
#          linewidth = 0.5, # 折线宽度
#          color = 'steelblue', # 折线颜色
#          marker = 'o', # 折线图中添加圆点
#          markersize = 3, # 点的大小
#          markeredgecolor='black', # 点的边框色
#          markerfacecolor='brown', # 点的填充色
#          label = 'invcost'
#          )
# fig.add_subplot(3,2,7)
# plt.plot(x_stocost, # x轴数据
#          y_stocost, # y轴数据
#          linestyle = '-', # 折线类型
#          linewidth = 0.5, # 折线宽度
#          color = 'steelblue', # 折线颜色
#          marker = 'o', # 折线图中添加圆点
#          markersize = 3, # 点的大小
#          markeredgecolor='black', # 点的边框色
#          markerfacecolor='brown', # 点的填充色
#          label = 'stocost'
#          )
# fig.add_subplot(3,2,8)
# plt.plot(x_roucost, # x轴数据
#          y_roucost, # y轴数据
#          linestyle = '-', # 折线类型
#          linewidth = 0.5, # 折线宽度
#          color = 'steelblue', # 折线颜色
#          marker = 'o', # 折线图中添加圆点
#          markersize = 3, # 点的大小
#          markeredgecolor='black', # 点的边框色
#          markerfacecolor='brown', # 点的填充色
#          label = 'roucost'
#          )
#对于X轴，只显示x中各个数对应的刻度值
# plt.xticks(fontsize=8, )  #改变x轴文字值的文字大小
# 添加y轴标签
# plt.ylabel('人数')
# 添加图形标题
# plt.title('每天微信文章阅读人数趋势')
# 显示图形
plt.show()
fig.savefig('折线图.png')

# book = xlsxwriter.Workbook('test.xlsx')
# sheet = book.add_worksheet('demo')
# sheet.insert_image('D4','折线图.png')

'##-------------------------------输出excel-----------------------------------##'
writer = pd.ExcelWriter('结果.xlsx')

BestSol_df = pd.DataFrame(BestSol)
BestSol_df.to_excel(writer, float_format='%.5f', sheet_name=u'policy')
I_df = pd.DataFrame(I_output)
I_df.to_excel(writer, float_format='%.5f', sheet_name=u'库存水平')
d_df = pd.DataFrame(d_output)
d_df.to_excel(writer, float_format='%.5f', sheet_name=u'交付量')

sheet = writer.book.add_worksheet('折线图')
sheet.insert_image('A1','折线图.png')
writer.save()

print(Phase1_2(BestSol)[0:5])
print(1)