import networkx as nx
import random

def Double_Edge_Target_Swap_MultiDiGraph(original_multidigraph_g):
    
    # create a numbered edgelist
    edgelist = {}
    counter = 0
    for e in set(original_multidigraph_g.edges(data=False)):
        edgelist[counter] = e # 1 : (A,B)
        counter+=1

    # create eventlist & numbered_eventlist
    eventlist = {}
    for e in list(edgelist.values()):   
        eventlist[(e[0], e[1])] = [] 
    for e in original_multidigraph_g.edges(data=True):
        eventlist[(e[0], e[1])].append(e[2]) # (A,B) : {events}
    numbered_eventlist = {}
    for k in edgelist.keys(): 
        numbered_eventlist[k] = eventlist[edgelist[k]] # 1 : {events}
    
    # create a edgelist copy
    edgelist_copy = edgelist.copy()

    # define variables
    target_swaps = 10*len(edgelist)
    done_swaps = 0
    loops = 0

    # while loop starts
    while done_swaps < target_swaps:

        for k in edgelist_copy.keys():

            # pick two edges
            edge1 = edgelist_copy[k]
            random_number = random.randint(0,len(edgelist_copy)-1)
            edge2 = edgelist_copy[random_number]
            # check no overlaps [exclude self-loops and multi-links]
            if edge1[0] in edge2 or edge1[1] in edge2:
                continue
            # swap
            edgelist_copy[k] = (edge1[0], edge2[1])
            edgelist_copy[random_number] = (edge2[0], edge1[1])
            done_swaps += 1

        loops += 1

    # null model eventlist
    eventlist_null_model = []
    for k in numbered_eventlist.keys():
        for event in numbered_eventlist[k]:
            edge = (edgelist_copy[k][0], edgelist_copy[k][1], event)
            eventlist_null_model.append(edge)
            
    # creation of randomised graph
    G = nx.MultiDiGraph()
    for e in eventlist_null_model:
        G.add_edge(u_for_edge=e[0], v_for_edge=e[1],
                weight=e[2]['weight'],
                epoch=e[2]['epoch'],
                #area=e[2]['area']
                )
    
    return G

def Double_Edge_Source_Swap_MultiDiGraph(original_multidigraph_g):
    
    # create a numbered edgelist
    edgelist = {}
    counter = 0
    for e in set(original_multidigraph_g.edges(data=False)):
        edgelist[counter] = e # 1 : (A,B)
        counter+=1

    # create eventlist & numbered_eventlist
    eventlist = {}
    for e in list(edgelist.values()):   
        eventlist[(e[0], e[1])] = [] 
    for e in original_multidigraph_g.edges(data=True):
        eventlist[(e[0], e[1])].append(e[2]) # (A,B) : {events}
    numbered_eventlist = {}
    for k in edgelist.keys(): 
        numbered_eventlist[k] = eventlist[edgelist[k]] # 1 : {events}
    
    # create a edgelist copy
    edgelist_copy = edgelist.copy()

    # define variables
    target_swaps = 10*len(edgelist)
    done_swaps = 0
    loops = 0

    # while loop starts
    while done_swaps < target_swaps:

        for k in edgelist_copy.keys():

            # pick two edges
            edge1 = edgelist_copy[k]
            random_number = random.randint(0,len(edgelist_copy)-1)
            edge2 = edgelist_copy[random_number]
            # check no overlaps [exclude self-loops and multi-links]
            if edge1[0] in edge2 or edge1[1] in edge2:
                continue
            # swap
            edgelist_copy[k] = (edge2[0], edge1[1])
            edgelist_copy[random_number] = (edge1[0], edge2[1])
            done_swaps += 1

        loops += 1

    # null model eventlist
    eventlist_null_model = []
    for k in numbered_eventlist.keys():
        for event in numbered_eventlist[k]:
            edge = (edgelist_copy[k][0], edgelist_copy[k][1], event)
            eventlist_null_model.append(edge)
            
    # creation of randomised graph
    G = nx.MultiDiGraph()
    for e in eventlist_null_model:
        G.add_edge(u_for_edge=e[0], v_for_edge=e[1],
                weight=e[2]['weight'],
                epoch=e[2]['epoch'],
                #area=e[2]['area']
                )
    
    return G

def Double_Edge_Random_Swap_MultiDiGraph(original_multidigraph_g):
    
    # create a numbered edgelist
    edgelist = {}
    counter = 0
    for e in set(original_multidigraph_g.edges(data=False)):
        edgelist[counter] = e # 1 : (A,B)
        counter+=1

    # create eventlist & numbered_eventlist
    eventlist = {}
    for e in list(edgelist.values()):   
        eventlist[(e[0], e[1])] = [] 
    for e in original_multidigraph_g.edges(data=True):
        eventlist[(e[0], e[1])].append(e[2]) # (A,B) : {events}
    numbered_eventlist = {}
    for k in edgelist.keys(): 
        numbered_eventlist[k] = eventlist[edgelist[k]] # 1 : {events}
    
    # create a edgelist copy
    edgelist_copy = edgelist.copy()

    # define variables
    target_swaps = 10*len(edgelist)
    done_swaps = 0
    loops = 0

    # while loop starts
    while done_swaps < target_swaps:

        for k in edgelist_copy.keys():

            # pick two edges
            edge1 = edgelist_copy[k]
            random_number = random.randint(0,len(edgelist_copy)-1)
            edge2 = edgelist_copy[random_number]
            # check no overlaps [exclude self-loops and multi-links]
            if edge1[0] in edge2 or edge1[1] in edge2:
                continue
            # random swap
            random_number_swap = random.randint(0,100)
            if random_number_swap < 50:
                # target swap
                edgelist_copy[k] = (edge1[0], edge2[1])
                edgelist_copy[random_number] = (edge2[0], edge1[1])
            elif random_number_swap > 50:
                # source swap
                edgelist_copy[k] = (edge2[0], edge1[1])
                edgelist_copy[random_number] = (edge1[0], edge2[1])
            done_swaps += 1

        loops += 1

    # null model eventlist
    eventlist_null_model = []
    for k in numbered_eventlist.keys():
        for event in numbered_eventlist[k]:
            edge = (edgelist_copy[k][0], edgelist_copy[k][1], event)
            eventlist_null_model.append(edge)
            
    # creation of randomised graph
    G = nx.MultiDiGraph()
    for e in eventlist_null_model:
        G.add_edge(u_for_edge=e[0], v_for_edge=e[1],
                weight=e[2]['weight'],
                epoch=e[2]['epoch'],
                #area=e[2]['area']
                )
    
    return G