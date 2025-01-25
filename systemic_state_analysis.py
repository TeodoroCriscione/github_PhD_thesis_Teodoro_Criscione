import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

#########################################################################
### network simplification and reduction 
#########################################################################

def aggregate_parallel_edges(network):
    # aggregate parallel edges
    edgelist = {(e[0], e[1]):0 for e in network.edges(data=True)}
    for e in network.edges(data=True):
        edgelist[(e[0], e[1])] += e[2]['weight']
    return edgelist

def solve_reciprocal_edges(network, edgelist):
    # solve reciprocal ties
    simpl_reciprocal_edges = {(e[0], e[1]):0 for e in network.edges(data=True)}
    edgelist_list = list(edgelist.items())
    while edgelist_list:
        e = edgelist_list[0]
        # does "e" has reciprocal?
        if network.has_edge(e[0][1], e[0][0]):
            rec_w_straight = edgelist[(e[0][0], e[0][1])]
            rec_w_inverse = edgelist[(e[0][1], e[0][0])]
            # compare the weights of the two
            if rec_w_straight > rec_w_inverse:
                simpl_reciprocal_edges[(e[0][0], e[0][1])] = rec_w_straight - rec_w_inverse
            elif rec_w_straight < rec_w_inverse:
                simpl_reciprocal_edges[(e[0][1], e[0][0])] = rec_w_inverse - rec_w_straight
            elif rec_w_straight == rec_w_inverse:
                simpl_reciprocal_edges[(e[0][0], e[0][1])] = 0
            edgelist_list.remove(((e[0][0],e[0][1]),e[1]))
            edgelist_list.remove(((e[0][1],e[0][0]),edgelist[(e[0][1],e[0][0])]))
        # if not, just keep it and move on
        else:
            simpl_reciprocal_edges[(e[0][0], e[0][1])] = e[1]
            edgelist_list.remove(((e[0][0],e[0][1]),e[1]))
    # outcome
    simpl_reciprocal_edges = {(e[0][0], e[0][1]):e[1] for e in simpl_reciprocal_edges.items() if e[1] > 0}
    return simpl_reciprocal_edges

def find_lwcc_graph(edgelist):
    subnetwork = nx.DiGraph()
    # create the network
    for e in edgelist.keys():
        subnetwork.add_edge(e[0], e[1], weight=edgelist[(e[0], e[1])])
    # make sure you pick the largest connected component
    lwcc_nodes = max(nx.weakly_connected_components(subnetwork), key=len)
    lwcc = nx.subgraph(subnetwork, lwcc_nodes)
    return lwcc

def simplify_and_get_lwcc(network):
     edgelist = aggregate_parallel_edges(network)
     edgelist = solve_reciprocal_edges(network, edgelist)
     lwcc = find_lwcc_graph(edgelist)
     return lwcc

#########################################################################
### economic network synergy
#########################################################################

def multipliers(graph_dictionary):
    disbursedP1 = 0
    disbursedP2 = 0
    disbursedP3 = 0
    # P1
    for n in graph_dictionary['P1'].nodes(data=True):
        disbursedP1 += float(n[1]['net_inj'])
    # P2
    for n in graph_dictionary['P2'].nodes(data=True):
        disbursedP2 += float(n[1]['net_inj'])
    disbursedP2 += disbursedP1
    # P3
    for n in graph_dictionary['P3'].nodes(data=True):
        disbursedP3 += float(n[1]['net_inj'])
    disbursedP3 += disbursedP2
    # output
    vol1 = graph_dictionary['P1'].size('weight')
    vol2 = graph_dictionary['P2'].size('weight')
    vol3 = graph_dictionary['P3'].size('weight')

    output_dict = {'P1':vol1/disbursedP1, 
                   'P2':(vol2)/(disbursedP2), 
                   'P3':(vol3)/(disbursedP3)}
    return output_dict

def add_disbursements(network, subnetwork):
    inj_node_attribute = nx.get_node_attributes(network, 'net_inj')
    inj_node_attribute_lwcc = {n:0 for n in subnetwork.nodes()}
    for n in subnetwork.nodes():
        inj_node_attribute_lwcc[n] = inj_node_attribute[n]
    nx.set_node_attributes(subnetwork, inj_node_attribute_lwcc, 'net_inj')
    return subnetwork

def circular_network_synergy(network):
    initial_volume = network.size('weight')
    largestWcomponent = nx.DiGraph()
    # define 'capacity'
    capacity = {(e[0], e[1]):int(round(e[2]['weight'])) for e in network.edges(data=True)}
    capacity =  {k: (1 if v == 0 else v) for k, v in capacity.items()}
    for e in capacity.keys():
        # capacity is rounded to avoid floating errors
        largestWcomponent.add_edge(e[0],e[1],capacity=int(round(capacity[(e[0],e[1])])))
    nx.set_edge_attributes(largestWcomponent, capacity, 'capacity')
    # define 'demand' based on the (rounded) capacity of edges
    demand = {n:0 for n in largestWcomponent.nodes()}
    for e in largestWcomponent.edges(data=True):
        demand[e[0]] -= e[2]['capacity']
        demand[e[1]] += e[2]['capacity']
    nx.set_node_attributes(largestWcomponent, demand, 'demand')
    # calculate flow
    flowDict = nx.min_cost_flow(largestWcomponent, demand='demand', capacity='capacity', weight=None)
    # simplified final edgelist lwcc
    simplified_lwcc = {}
    for k in flowDict.keys():
        for j in flowDict[k].keys():
            if flowDict[k][j] > 0:
                simplified_lwcc[(k,j)] = flowDict[k][j]
    # create graph
    simplified_lwcc_graph = nx.DiGraph()
    for e in simplified_lwcc.keys():
        simplified_lwcc_graph.add_edge(e[0], e[1], weight=simplified_lwcc[(e[0],e[1])])
    # calculate circular network synergy
    simple_graph_volume = simplified_lwcc_graph.size('weight')
    CNS = 1-(simple_graph_volume/initial_volume)
    return CNS

#########################################################################
### ecological network analysis (ENA): systemic sustainability
#########################################################################

def no_float_tranform(network):
    edgelist = {(e[0], e[1]):int(round(e[2]['weight'])) for e in network.edges(data=True)}
    edgelist =  {k: (1 if v == 0 else v) for k, v in edgelist.items()}
    edgelist_status = nx.get_edge_attributes(network, 'edge_status')
    nodelist_status = nx.get_node_attributes(network, 'node_status')
    nodelist_inject = nx.get_node_attributes(network, 'net_inj')
    nodelist_inject = {n:int(round(float(v))) for n,v in nodelist_inject.items()}
    nodelist_inject =  {k: (1 if v == 0 else v) for k, v in nodelist_inject.items()}
    n_network = nx.DiGraph()
    for e in edgelist.keys():
        n_network.add_edge(e[0], e[1], weight=edgelist[(e[0], e[1])])
    nx.set_edge_attributes(n_network, edgelist_status, 'edge_status')
    nx.set_node_attributes(n_network, nodelist_status, 'node_status')
    nx.set_node_attributes(n_network, nodelist_inject, 'net_inj')
    return n_network

def ena_node_flow_calculator(network):
    node_inflow = {n:0 for n in network.nodes()}
    node_outflow = {n:0 for n in network.nodes()}
    for e in network.edges(data=True):
        node_inflow[e[1]] += e[2]['weight']
        node_outflow[e[0]] += e[2]['weight']
    return node_inflow, node_outflow

def ena_systemic_metrics_log2(network):
    network = no_float_tranform(network)
    node_inflow, node_outflow = ena_node_flow_calculator(network)
    volume = round(network.size('weight'), 2)
    ascendency_results = []
    reserve_results = []
    capacity_results = []
    phi_results = []

    effective_flows_results = []
    effective_nodes_results = []
    effective_roles_results = []

    for e in network.edges(data=True):
        w = round(e[2]['weight'], 2)

        edge_ascendency = w * np.log2((w*volume)/(node_outflow[e[0]]*node_inflow[e[1]]))
        edge_reserve = w * np.log2((w**2)/(node_outflow[e[0]]*node_inflow[e[1]]))
        edge_capacity = w * np.log2(w/volume)
        edge_phi = ((w**2)/(node_outflow[e[0]]*node_inflow[e[1]]))**((-1)*(1/2)*(w/volume))

        flow_eff = (w/volume)**((-1)*(w/volume))
        node_eff = ((volume**2/(node_inflow[e[1]]*node_outflow[e[0]])))**((1/2)*(w/volume))
        role_eff = ((w*volume)/(node_outflow[e[0]]*node_inflow[e[1]]))**(w/volume)

        ascendency_results.append(edge_ascendency)
        reserve_results.append(edge_reserve)
        capacity_results.append(edge_capacity)
        phi_results.append(edge_phi)

        effective_flows_results.append(flow_eff)
        effective_nodes_results.append(node_eff)
        effective_roles_results.append(role_eff)
    #
    ascendency = round(sum(ascendency_results), 2)
    reserve = round((-1)*sum(reserve_results), 2)
    capacity = round((-1)*sum(capacity_results), 2)

    return ascendency, reserve, capacity

####################################################################
######## Synergy ########
####################################################################

def write_results_systemic_state(results, output_folder):
    f = open(output_folder+'/systemic_state.txt', "a")
    for r in results:
        f.write(str(r))
        f.write(';')
    f.write('\n')
    f.close()