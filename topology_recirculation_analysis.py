# required packages
import networkx as nx
import numpy as np
from collections import Counter
import datetime
import pandas as pd
import os

####################################
##### SET 1: generic functions #####
####################################

# convert to DiGraph
def conversion_to_digraph(net):
    business_type_node = nx.get_node_attributes(net, name='business')
    area_type_node = nx.get_node_attributes(net, name='area')
    top_status_type_node = nx.get_node_attributes(net, name='node_status')
    net_status_type_node = nx.get_node_attributes(net, name='netflow_status')

    # convert to DiGraph
    edgelist = {(e[0], e[1]):0 for e in net.edges()}
    for e in net.edges(data=True):
        edgelist[(e[0], e[1])] += e[2]['weight']
    dnet = nx.DiGraph()
    for e in edgelist.keys():
        dnet.add_edge(e[0], e[1], weight=edgelist[(e[0], e[1])])

    nx.set_node_attributes(dnet, business_type_node, 'business')
    nx.set_node_attributes(dnet, area_type_node, 'area')
    nx.set_node_attributes(dnet, top_status_type_node, 'node_status')
    nx.set_node_attributes(dnet, net_status_type_node, 'netflow_status')
    return dnet

def convert_time(seconds, granularity=2):
    # from https://stackoverflow.com/questions/4048651/function-to-convert-seconds-into-minutes-hours-and-days
    intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1))
    result = []
    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])

def export_edgelist_with_data(net):
    edges_with_data = {}
    for e in net.edges():
        edges_with_data[(e[0],e[1])] = []
    for e in net.edges(data=True):
        edges_with_data[(e[0],e[1])].append(e[2])
    return edges_with_data
    
def export_nodelist_with_data(net, _node_status_):
    # add edge data
    nodes_with_data = {n:[] for n in net.nodes()}
    for e in sorted(net.edges(data=True), key=lambda x: x[2]['epoch']):
        nodes_with_data[e[0]].append(tuple(['outgoing', e[1], _node_status_[e[1]], e[2]]))
        nodes_with_data[e[1]].append(tuple(['incoming', e[0], _node_status_[e[0]], e[2]]))
    # sort operations
    for n in nodes_with_data.keys():
        nodes_with_data[n] = sorted(nodes_with_data[n], key=lambda x: x[3]['epoch'])
    return nodes_with_data

#######################################################################
################ SET2: topological assignment #########################
#######################################################################

def write_results_topology_summary(results, graph_name, output_folder):
    f = open(output_folder+'/topology_summary.txt', "a")
    f.write(graph_name)
    f.write(';')
    for r in results:
        f.write(str(r))
        f.write(';')
    f.write('\n')
    f.close()

def write_results_topological_analysis_summary(results, output_folder):
    f = open(output_folder+'/topological_analysis_summary.txt', "a")
    for r in results:
        f.write(str(r))
        f.write(';')
    f.write('\n')
    f.close()

def topological_features_summary(net):
    dnet = conversion_to_digraph(net)
    # count scc
    counter_scc = 0
    nodes_in_scc = []
    for s in nx.strongly_connected_components(dnet):
        if len(s) > 1:
            counter_scc += 1
            nodes_in_scc = nodes_in_scc + list(s)
    # count wcc
    counter_wcc = 0
    for w in nx.weakly_connected_components(dnet):
        if len(w) > 1: counter_wcc += 1
    # analysis of weights and time
    weights = []
    timestamps = []
    for e in net.edges(data=True):
        weights.append(e[2]['weight'])
        timestamps.append(e[2]['epoch'])
    timestamps = [float(t) for t in timestamps]
    _results_ = [counter_scc, counter_wcc, net.number_of_nodes(), 
                 dnet.number_of_edges(), net.number_of_edges(),  net.size('weight'),
                  np.mean(weights), np.std(weights), min(weights), max(weights),
                  datetime.datetime.fromtimestamp(min(timestamps)),
                  datetime.datetime.fromtimestamp(max(timestamps)),
                  max(timestamps)-min(timestamps),
                  convert_time(max(timestamps)-min(timestamps))]
    return nodes_in_scc, _results_

def node_edge_status_assignment(net):

    # convert into directed
    dnet = conversion_to_digraph(net)
    component_node_status_dict = {n:0 for n in dnet.nodes()}
    component_edge_status_dict =  {e:0 for e in dnet.edges()}

    # sccs # S^(Ns)
    for s in nx.strongly_connected_components(dnet):
        if len(s) > 1:
            for n in s:
                component_node_status_dict[n] = 'scc'
        else: 
            # nodes not in sccs are reported as single nodes
            continue 

    # edge status (first categorisation) # + Em
    for e in net.edges():
        if (component_node_status_dict[e[0]] == 'scc') and (component_node_status_dict[e[1]] == 'scc'):
            component_edge_status_dict[e] = 'edge_scc'
        elif (component_node_status_dict[e[0]] == 0) and (component_node_status_dict[e[1]] == 0):
            component_edge_status_dict[e] = 'edge_dag'
        elif (component_node_status_dict[e[0]] == 0) and (component_node_status_dict[e[1]] == 'scc'):
            component_edge_status_dict[e] = 'edge_from_outscc_to_scc'
        elif (component_node_status_dict[e[0]] == 'scc') and (component_node_status_dict[e[1]] == 0):
            component_edge_status_dict[e] = 'edge_from_scc_to_outscc'
    nx.set_edge_attributes(dnet, component_edge_status_dict, 'edge_status')

    # set of type of edge incoming to / outgoing from each node 
    incoming_type_x_node = {n:set() for n in dnet.nodes()}
    outgoing_type_x_node = {n:set() for n in dnet.nodes()}
    for e in dnet.edges(data=True): # + Ed
        edge_type = e[2]['edge_status']
        outgoing_type_x_node[e[0]].add(edge_type)
        incoming_type_x_node[e[1]].add(edge_type)

    # isolate dags and tendrils to classify them
    subgraph_dags_tendrils = dnet.subgraph([n for n,v in component_node_status_dict.items() if v == 0]) # exclude sccs nodes
    for w in nx.weakly_connected_components(subgraph_dags_tendrils): # +2W^(Nw)
        w = list(w)
        if len(w) == 1: 
            # tendrils
            if outgoing_type_x_node[w[0]] == set() and incoming_type_x_node[w[0]] != set():
                component_node_status_dict[w[0]] = 'out_tendril'
            elif outgoing_type_x_node[w[0]] != set() and incoming_type_x_node[w[0]] == set():
                component_node_status_dict[w[0]] = 'in_tendril'
            # bridges
            elif outgoing_type_x_node[w[0]] == {'edge_from_outscc_to_scc'} and incoming_type_x_node[w[0]] == {'edge_from_scc_to_outscc'}:
                component_node_status_dict[w[0]] = 'bridge_scc'
        elif len(w) > 1: # if they are dags
            edgeset_type_w = []
            for n in w:
                edgeset_type_w = edgeset_type_w + list(outgoing_type_x_node[n])
                edgeset_type_w = edgeset_type_w + list(incoming_type_x_node[n])
            edgeset_type_w = set(edgeset_type_w)
            if edgeset_type_w == {'edge_dag'}:
                for n in w:
                    component_node_status_dict[n] = 'dag0'
            elif 'edge_from_scc_to_outscc' in edgeset_type_w and 'edge_from_outscc_to_scc' not in edgeset_type_w:
                for n in w:
                    component_node_status_dict[n] = 'dagTout'
            elif 'edge_from_outscc_to_scc' in edgeset_type_w and 'edge_from_scc_to_outscc' not in edgeset_type_w:
                for n in w:
                    component_node_status_dict[n] = 'dagTin'
            elif  'edge_from_scc_to_outscc' in edgeset_type_w and 'edge_from_outscc_to_scc' in edgeset_type_w:
                for n in w:
                    component_node_status_dict[n] = 'dagTmix'

    # classify scc
    subgraphs_of_sccs = [] 
    for s in nx.strongly_connected_components(dnet): # 2S^(Ns)
        s = list(s)
        subgraphs_of_sccs.append(dnet.subgraph(s))
        if len(s) > 1:
            edgeset_type_s = []
            for n in s:
                edgeset_type_s = edgeset_type_s + list(outgoing_type_x_node[n])
                edgeset_type_s = edgeset_type_s + list(incoming_type_x_node[n])
            edgeset_type_s = set(edgeset_type_s)
            if edgeset_type_s == {'edge_scc'}:
                for n in s:
                    component_node_status_dict[n] = 'scc0'
            elif 'edge_from_scc_to_outscc' in edgeset_type_s and 'edge_from_outscc_to_scc' not in edgeset_type_s:
                for n in s:
                    component_node_status_dict[n] = 'sccTout'
            elif 'edge_from_outscc_to_scc' in edgeset_type_s and 'edge_from_scc_to_outscc' not in edgeset_type_s:
                for n in s:
                    component_node_status_dict[n] = 'sccTin'
            elif 'edge_from_scc_to_outscc' in edgeset_type_s and 'edge_from_outscc_to_scc' in edgeset_type_s:
                for n in s:
                    component_node_status_dict[n] = 'sccTmix'
        else:
            continue

    # edges within sccs
    edges_in_subgraphs_sccs_strict = []
    for x in subgraphs_of_sccs:
        edges_in_subgraphs_sccs_strict = edges_in_subgraphs_sccs_strict + list(x.edges())

    ############### SECOND PART #####################
        
    component_edge_status_dict = {e:0 for e in dnet.edges()}
    # edge second categorisation
    for e in component_edge_status_dict.keys(): #+E

        # scc
        if component_node_status_dict[e[0]] == 'scc0' and component_node_status_dict[e[1]] == 'scc0':
            component_edge_status_dict[e] = 'edge_scc0'
        elif component_node_status_dict[e[0]] == 'sccTin' and component_node_status_dict[e[1]] == 'sccTin':  
            component_edge_status_dict[e] = 'edge_sccTin'
        elif component_node_status_dict[e[0]] == 'sccTout' and component_node_status_dict[e[1]] == 'sccTout':     
            component_edge_status_dict[e] = 'edge_sccTout'
        elif component_node_status_dict[e[0]] == 'sccTmix' and component_node_status_dict[e[1]] == 'sccTmix':      
            component_edge_status_dict[e] = 'edge_sccTmix'

        # dag
        elif component_node_status_dict[e[0]] == 'dag0' and component_node_status_dict[e[1]] == 'dag0':
            component_edge_status_dict[e] = 'edge_dag0'
        elif component_node_status_dict[e[0]] == 'dagTin' and component_node_status_dict[e[1]] == 'dagTin':  
            component_edge_status_dict[e] = 'edge_dagTin'
        elif component_node_status_dict[e[0]] == 'dagTout' and component_node_status_dict[e[1]] == 'dagTout':     
            component_edge_status_dict[e] = 'edge_dagTout'
        elif component_node_status_dict[e[0]] == 'dagTmix' and component_node_status_dict[e[1]] == 'dagTmix':      
            component_edge_status_dict[e] = 'edge_dagTmix'

        # dag2scc or scc2dag
        elif component_node_status_dict[e[0]] in ['dagTin', 'dagTmix'] and component_node_status_dict[e[1]] in ['sccTin', 'sccTmix']:
            component_edge_status_dict[e] = 'edge_dag2scc'
        elif component_node_status_dict[e[0]] in ['sccTout', 'sccTmix'] and component_node_status_dict[e[1]] in ['dagTout', 'dagTmix']:
            component_edge_status_dict[e] = 'edge_scc2dag'
        
        # tendrils
        elif component_node_status_dict[e[0]] == 'in_tendril' and component_node_status_dict[e[1]] in ['sccTin', 'sccTmix']:
            component_edge_status_dict[e] = 'edge_in_tendril'
        elif component_node_status_dict[e[0]] in ['sccTout', 'sccTmix'] and component_node_status_dict[e[1]] == 'out_tendril':
            component_edge_status_dict[e] = 'edge_out_tendril'

        # bridge
        elif component_node_status_dict[e[0]] == 'bridge_scc' and component_node_status_dict[e[1]] in ['scc0','sccTin','sccTout','sccTmix']:      
            component_edge_status_dict[e] = 'edge_bridge_scc'
        elif component_node_status_dict[e[0]] in ['scc0','sccTin','sccTout','sccTmix'] and component_node_status_dict[e[1]] == 'bridge_scc':      
            component_edge_status_dict[e] = 'edge_bridge_scc'

    edges_in_subgraphs_sccs_large = [k for k,v in component_edge_status_dict.items() if v in ['edge_scc0', 'edge_sccTin', 'edge_sccTout', 'edge_sccTmix', 0]] 
    edges_in_scc_left = set(edges_in_subgraphs_sccs_large).difference(set(edges_in_subgraphs_sccs_strict))
    for e in edges_in_scc_left:
        component_edge_status_dict[e] = 'edge_scc2scc'

    return component_node_status_dict, component_edge_status_dict

def topological_categories_analysis(network):
    component_node_status_dict, component_edge_status_dict = node_edge_status_assignment(network)
    dnet = conversion_to_digraph(network)
    # count node status
    counter_node_status_dict = dict(Counter(component_node_status_dict.values()))
    # count edge status as dlinks
    counter_edge_status_dict = dict(Counter(component_edge_status_dict.values()))

    # list of operations per edge
    edges_with_data = {e:[] for e in dnet.edges()}
    for e in network.edges(data=True): # + E
        edges_with_data[(e[0], e[1])].append(e)

    # list of operations per group
    topological_edge_grouping = {k:[] for k in list(counter_edge_status_dict.keys())}
    for e in edges_with_data.keys():
        edge_status = component_edge_status_dict[(e[0], e[1])]
        topological_edge_grouping[edge_status] = topological_edge_grouping[edge_status] + edges_with_data[(e[0], e[1])]

    # edge analysis
    summary_topological_edge_group = {k:[] for k in list(counter_edge_status_dict.keys())}
    for k in topological_edge_grouping.keys(): # T^(S+W)
        weights = [e[2]['weight'] for e in topological_edge_grouping[k]]
        timestamps = [float(e[2]['epoch']) for e in topological_edge_grouping[k]]
        subgraph = nx.from_edgelist(topological_edge_grouping[k], create_using=nx.MultiDiGraph())
        dsubgraph = conversion_to_digraph(subgraph)
        counter_scc = 0
        for s in nx.strongly_connected_components(dsubgraph ):
            if len(s) > 1: counter_scc += 1
            else: continue
        counter_wcc = 0
        for w in nx.weakly_connected_components(dsubgraph ):
            if len(w) > 1: counter_wcc += 1
            else: continue
        summary_topological_edge_group [k] = [counter_scc, counter_wcc, 
                                            dsubgraph.number_of_nodes(), 
                                            dsubgraph.number_of_edges(), 
                                            subgraph.number_of_edges(),
                                            sum(weights), 
                                            np.mean(weights), np.std(weights), 
                                            min(weights), max(weights),
                                            min(timestamps), max(timestamps),
                                            max(timestamps)-min(timestamps)]

    return summary_topological_edge_group

def single_tx_nodes(nodes_with_data, _node_status_):
    # nodes with only one transaction
    one_tx_nodes = {}
    one_tx_nodes['incoming'] = []
    one_tx_nodes['outgoing'] = []
    for n in nodes_with_data.keys():
        if len(nodes_with_data[n])==1:
            if nodes_with_data[n][0][0] == 'incoming':
                one_tx_nodes['incoming'].append(tuple([n, _node_status_[n]]))
            elif nodes_with_data[n][0][0] == 'outgoing':
                one_tx_nodes['outgoing'].append(tuple([n, _node_status_[n]]))
        else:
            continue
    return one_tx_nodes

#####################################################################
################ SET 3: recirculation dynamics #########################
#####################################################################

def recirculation_operations_edgelist(nodes_with_data, net):
    recirc_ops = {}
    recirc_stats = {}
    for n in net.nodes():
        recirc_ops[n] = {}
        recirc_stats[n] = {}
        ops = nodes_with_data[n] # list operations by node n
        inc_ops = []
        outg_ops = []
        counter = 0
        for e in ops:
            if e[0]=='incoming' and outg_ops == []:
                inc_ops.append(e)
            elif e[0] == 'outgoing' and outg_ops == [] and inc_ops != []: # got outgoing with already an incoming
                outg_ops.append(e)
            elif e[0]=='incoming' and outg_ops != []:
                counter += 1
                # store edges
                recirc_ops[n][counter] = inc_ops + outg_ops
                # re-circulation time: max(last outg - first inc)
                re_circ_diff_t_max = float(outg_ops[-1][3]['epoch']) - float(inc_ops[0][3]['epoch']) 
                # re-circulation time: min(last outg - first inc)
                re_circ_diff_t_min = float(outg_ops[0][3]['epoch']) - float(inc_ops[-1][3]['epoch'])
                # store diff
                recirc_stats[n][counter] = [re_circ_diff_t_min, re_circ_diff_t_max]
                # reset
                inc_ops = []
                outg_ops = []
                inc_ops.append(e)
    return recirc_ops, recirc_stats

def create_dictionary_recirculation_operations(recircops, recircstats):
    recirculation_edgelist = {}
    # initialise
    for n in recircops.keys():
        for c in recircops[n].keys():
            for op in recircops[n][c]:
                recirculation_edgelist[tuple([n, op[1]])] = []
                recirculation_edgelist[tuple([op[1], n])] = []
    # populate
    for n in recircops.keys():
        for c in recircops[n].keys():
            for op in recircops[n][c]:
                _recirc_edge_ = [op[3]['weight'], op[3]['epoch'],
                                 op[3]['edge_status'],n, recircstats[n][c][0], recircstats[n][c][1]]
                if op[0] == 'outgoing' and _recirc_edge_ not in recirculation_edgelist[tuple([n, op[1]])]:
                    recirculation_edgelist[tuple([n, op[1]])].append(tuple(_recirc_edge_))
                elif op[0] == 'incoming' and _recirc_edge_ not in recirculation_edgelist[tuple([op[1], n])]:
                    recirculation_edgelist[tuple([op[1], n])].append(tuple(_recirc_edge_))
    return recirculation_edgelist

def recirculation_dataframe_operations(recirculation_edgelist):
    selected_edgelist = set()
    for e in recirculation_edgelist.keys():
        for i in recirculation_edgelist[e]:
            _edge_ = tuple([e[0], e[1], i[0], i[1], i[2], i[3], i[4], i[5]]) #, i[6]
            selected_edgelist.add(_edge_)
    # every transaction can be part of more than one operation
    return pd.DataFrame(selected_edgelist, columns=['source', 'target', 'weight',
                                                    'epoch', 'edge_status', 'recirc_user', 'rtmin', 'rtmax'])

def recirculation_dataframe_users(recirc_stats):
    recirc_times = []
    for n in recirc_stats.keys():
        for c in recirc_stats[n]:
            i = recirc_stats[n][c]
            recirc_times.append(tuple([n, c, i[0], i[1]]))
    # COUNT is the operation number for that particular user
    # rtmin is the min-time of that operation
    # rtmax is the max-time of that operation
    # e.g. operation COUNT 2 is the second operation for that user
    return  pd.DataFrame(recirc_times, columns=['ID', 'COUNT', 'rtmin', 'rtmax'])

def add_temporal_categories(recirc_users, recirc_tx):
    counter_seconds = counter_function(dataframe=recirc_users, 
                                        col_dataframe='rtmax')
    lst = distribution_metrics(distribution_frequency=counter_seconds, 
                            dataframe=recirc_users, 
                            col_dataframe='rtmax')
    moda = lst[0]
    q1_d = lst[1]
    q2_d = lst[2]
    q3_d = lst[3]
    recirc_users['FREQ'] = recirc_users.rtmax.apply(lambda x: 'HFQ1' if (x <= q1_d[0]) else \
                        'HFQ2' if ((x > q1_d[0]) & (x<= q2_d[0])) else \
                        'HFQ3' if ((x > q2_d[0]) & (x<= q3_d[0])) else \
                        'LFQ3' if (x > q3_d[0]) else 0)
    recirc_tx['FREQ'] = recirc_tx.rtmax.apply(lambda x: 'HFQ1' if (x <= q1_d[0]) else \
                        'HFQ2' if ((x > q1_d[0]) & (x<= q2_d[0])) else \
                        'HFQ3' if ((x > q2_d[0]) & (x<= q3_d[0])) else \
                        'LFQ3' if (x > q3_d[0]) else 0)
    return recirc_users, recirc_tx

def dataframe_creation(recirc_tx, _node_status_, _edge_status_):

    recirculating_users = {} # unique recirculating users
    recirculating_tx = {} # unique recirculating transactions
    for i in recirc_tx.iterrows():
        recirculating_users[i[1].recirc_user] = {'HFQ1':0,'HFQ2':0,'HFQ3':0,'LFQ3':0}
        recirculating_tx[tuple([i[1].source, i[1].target, i[1].weight, i[1].epoch])] = {'HFQ1':0,'HFQ2':0,'HFQ3':0,'LFQ3':0}
    for i in recirc_tx.iterrows():
        recirculating_users[i[1].recirc_user][i[1].FREQ] += 1 
        recirculating_tx[tuple([i[1].source, i[1].target, i[1].weight, i[1].epoch])][i[1].FREQ] += 1
    
    # recirculating users dataframe
    recirculating_users_data = pd.DataFrame.from_dict(recirculating_users, orient='index')
    #node_status_assignment = nx.get_node_attributes(graph, 'node_status')
    recirculating_users_data = pd.DataFrame.from_dict(recirculating_users, orient='index')
    recirculating_users_data['ID'] = recirculating_users_data.index
    recirculating_users_data['node_status'] = recirculating_users_data.ID.apply(lambda x: _node_status_[x])
    recirculating_users_data['TOT'] = recirculating_users_data.HFQ1 + recirculating_users_data.HFQ2 + \
                                    recirculating_users_data.HFQ3 + recirculating_users_data.LFQ3
    # recirculating tx dataframe
    recirculating_tx_data = pd.DataFrame.from_dict(recirculating_tx, orient='index')
    recirculating_tx_data.reset_index(inplace=True)
    recirculating_tx_data.columns = ['source', 'target', 'weight', 'epoch', 'HFQ1', 'HFQ2', 'HFQ3', 'LFQ3']
    recirculating_tx_data['edge_status'] = recirculating_tx_data.apply(lambda x: _edge_status_[(x.source,x.target)], axis=1)
    recirculating_tx_data['TOT'] = recirculating_tx_data.HFQ1 + recirculating_tx_data.HFQ2 + \
                                            recirculating_tx_data.HFQ3 + recirculating_tx_data.LFQ3
    return recirculating_users_data, recirculating_tx_data

def recirculation_analysis(recirc_users, recirc_tx, model_type, model_number, period):
    #add period
    recirc_users[period] = period
    recirc_tx[period] = period
    
    # count ONLY recirculating users per each node_status
    recirc_users_freq_list = []
    for fq in ['HFQ1', 'HFQ2', 'HFQ3', 'LFQ3']:
        recirc_users_freq  = recirc_users.groupby(['node_status'])[fq].sum().reset_index()
        recirc_users_freq ['FREQ'] = fq
        recirc_users_freq ['period'] = period
        recirc_users_freq ['model_type'] = model_type
        recirc_users_freq ['model_number'] = model_number
        recirc_users_freq ['value'] = recirc_users_freq[fq]
        recirc_users_freq .drop(columns=[fq], inplace=True)
        recirc_users_freq_list.append(recirc_users_freq)
    recirc_users_dataframe = pd.concat(recirc_users_freq_list)

    # count transactions per each edge_status
    recirc_tx_freq_list = []
    for fq in ['HFQ1', 'HFQ2', 'HFQ3', 'LFQ3']:
            recirc_tx_freq  = recirc_tx.groupby(['edge_status'])[fq].sum().reset_index()
            recirc_tx_freq ['FREQ'] = fq
            recirc_tx_freq ['period'] = period
            recirc_tx_freq ['model_type'] = model_type
            recirc_tx_freq ['model_number'] = model_number
            recirc_tx_freq ['value'] = recirc_tx_freq [fq]
            recirc_tx_freq .drop(columns=[fq], inplace=True)
            recirc_tx_freq_list.append(recirc_tx_freq )
    recirc_tx_dataframe = pd.concat(recirc_tx_freq_list)
    recirc_tx_dataframe['edge_status'] = recirc_tx_dataframe.edge_status.apply(lambda x: 'edge_in_single_node' if x=='edge_in_tendril' else \
                                                                            'edge_out_single_node' if x =='edge_out_tendril' else x)
    # weighting
    for fq in ['HFQ1', 'HFQ2', 'HFQ3', 'LFQ3']:
         recirc_tx['w'+fq] = recirc_tx[fq]*(recirc_tx['weight'] / recirc_tx['TOT'])

    # sum 'corrected' weight transactions per each edge_status
    recirc_wtx_freq_list = []
    for fq in ['wHFQ1', 'wHFQ2', 'wHFQ3', 'wLFQ3']:
            recirc_wtx_freq  = recirc_tx.groupby(['edge_status'])[fq].sum().reset_index()
            recirc_wtx_freq ['FREQ'] = fq
            recirc_wtx_freq ['period'] = period
            recirc_wtx_freq ['model_type'] = model_type
            recirc_wtx_freq ['model_number'] = model_number
            recirc_wtx_freq ['value'] = recirc_wtx_freq [fq]
            recirc_wtx_freq .drop(columns=[fq], inplace=True)
            recirc_wtx_freq_list.append(recirc_wtx_freq )
    recirc_wtx_dataframe = pd.concat(recirc_wtx_freq_list)
    recirc_wtx_dataframe['edge_status'] = recirc_wtx_dataframe.edge_status.apply(lambda x: 'edge_in_single_node' if x=='edge_in_tendril' else \
                                                                            'edge_out_single_node' if x =='edge_out_tendril' else x)
    
    return recirc_users_dataframe, recirc_tx_dataframe, recirc_wtx_dataframe

def export_temporal_recirculation_metrics(network, output_folder, model_type, model_number, period):

    # extract
    _node_status_, _edge_status_ = node_edge_status_assignment(network)
    nodes_with_data = export_nodelist_with_data(network, _node_status_)
    recirc_ops, recirc_stats = recirculation_operations_edgelist(nodes_with_data, network)
    recirculation_edgelist = create_dictionary_recirculation_operations(recirc_ops, recirc_stats)
    
    # data frame operations
    recirc_tx = recirculation_dataframe_operations(recirculation_edgelist)
    recirc_tx= recirc_tx[['source', 'target', 'weight', 'epoch', 'edge_status','recirc_user', 'rtmin', 'rtmax']]
    recirc_tx, dyad_counter = detect_prominent_dyads_operations(recirc_tx)
    
    # data frame users
    recirc_users = recirculation_dataframe_users(recirc_stats)
    recirc_users.rename(columns={'MIN':'rtmin', 'MAX':'rtmax'}, inplace=True)
    recirc_users = recirc_users[['ID', 'COUNT', 'rtmin', 'rtmax']] # COUNT is the operation number for that particular user
    recirc_users['index'] = recirc_users.index

    # add temporal categories
    recirc_users, recirc_tx = add_temporal_categories(recirc_users, recirc_tx)
    recirculating_users_data, recirculating_tx_data = dataframe_creation(recirc_tx, _node_status_, _edge_status_)
    recirc_users_dataframe, recirc_tx_dataframe, recirc_wtx_dataframe = recirculation_analysis(recirculating_users_data, 
                                                                                               recirculating_tx_data, 
                                                                                               model_type, 
                                                                                               model_number, 
                                                                                               period)
    path = output_folder+'recirculation_significance/'
    os.makedirs(path, exist_ok=True)
    recirc_users_dataframe.to_csv(path+'recirc_users_'+model_type+'_'+str(model_number)+'_'+period+'.csv')
    recirc_tx_dataframe.to_csv(path+'recirc_tx_'+model_type+'_'+str(model_number)+'_'+period+'.csv')
    recirc_wtx_dataframe.to_csv(path+'recirc_wtx_'+model_type+'_'+str(model_number)+'_'+period+'.csv')

def export_recirculation_files(net, graph_name, output_folder):
    # extract
    _node_status_, _edge_status_ = node_edge_status_assignment(net)
    nodes_with_data = export_nodelist_with_data(net, _node_status_)
    recirc_ops, recirc_stats = recirculation_operations_edgelist(nodes_with_data, net)
    recirculation_edgelist = create_dictionary_recirculation_operations(recirc_ops, recirc_stats)
    
    # write data frame operations
    recirc_df_operations = recirculation_dataframe_operations(recirculation_edgelist)
    path = output_folder+'/recirculation_metrics/'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    recirc_df_operations.to_csv(path+'/'+graph_name+"_recirc_operations.csv")
    
    # write data frame users
    recirc_df_users = recirculation_dataframe_users(recirc_stats)
    path = output_folder+'/recirculation_metrics/'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    recirc_df_users.to_csv(path+'/'+graph_name+"_recirc_users.csv")


def edgeset_dataframe(transaction_dataframe, net):
    edgeset = set()
    for e in transaction_dataframe.iterrows():
        _edge_ = tuple([e[1].source, e[1].target, e[1].weight, e[1].epoch, e[1].transaction_prom, e[1].dyad_prom])
        edgeset.add(_edge_)
    edgeset_df = pd.DataFrame(edgeset, columns=['source', 'target', 'weight', 'epoch', 'transaction_prom', 'dyad_prom'])
    edgeset_df['epoch'] = edgeset_df.epoch.astype(str)
    # merge edgeset (no repeating) with total edgelist dataframe
    edgelist_g = nx.to_pandas_edgelist(net, edge_key="ekey")
    # every transaction is taken only once even if part of more operations
    return pd.merge(edgeset_df, edgelist_g, how='left', 
                                on=['source', 'target', 'weight', 'epoch'])

def detect_prominent_edges(transaction_dataframe):
    edge_counter = {(e[1].source, e[1].target):0 for e in transaction_dataframe.iterrows()}
    for e in transaction_dataframe.iterrows():
        edge_counter[(e[1].source, e[1].target)]+=1
    prominent_edges = {k:v for k,v in edge_counter.items() if v>1}
    return prominent_edges

def detect_prominent_dyads_operations(transaction_dataframe):
    transaction_counter = {tuple([e[1].source, e[1].target, str(e[1].weight), str(e[1].epoch)]):0 for e in transaction_dataframe.iterrows()}
    dyad_counter = {tuple([e[1].source, e[1].target]):0 for e in transaction_dataframe.iterrows()}
    for e in transaction_dataframe.iterrows():
        transaction_counter[tuple([e[1].source, e[1].target, str(e[1].weight), str(e[1].epoch)])] += 1
        dyad_counter[tuple([e[1].source, e[1].target])] += 1
    # add to original dataframe
    transaction_prom_lst_values = []
    dyad_prom_lst_values = []
    for i in transaction_dataframe.iterrows():
        transaction_prom_lst_values.append(transaction_counter[tuple([i[1].source, i[1].target, 
                                    str(i[1].weight), str(i[1].epoch)])]) 
        dyad_prom_lst_values.append(dyad_counter[tuple([e[1].source, e[1].target])])
    transaction_dataframe['transaction_prom'] = transaction_prom_lst_values
    transaction_dataframe['dyad_prom'] = dyad_prom_lst_values
    return transaction_dataframe, dyad_counter

def detect_prominent_nodes(net, users_dataframe):
    nodeattdf = pd.DataFrame.from_dict(dict(net.nodes(data=True)), orient='index')
    nodeattdf['ID'] = nodeattdf.index
    users_dataframe = pd.merge(users_dataframe, nodeattdf, how='left', on='ID')
    # prominence
    user_prominence = {n:0 for n in net.nodes()}
    for n in users_dataframe.ID:
        user_prominence[n] += 1
    user_prominence = {k:v for k,v in user_prominence.items() if v>0}
    users_dataframe['node_prom'] = users_dataframe.ID.apply(lambda x: user_prominence[x])
    users_dataframe.sort_values(by='node_prom', ascending=False, inplace=True)
    return users_dataframe

def counter_function(dataframe, col_dataframe):
        counter_seconds = {}
        # count how many times that value (in seconds) repeated # round(float(s),2)
        counter_seconds = {s:0 for s in set(dataframe[col_dataframe])} # keys are recirc times
        for s in dataframe[col_dataframe]:
            counter_seconds[s] += 1 # values are multiplicities of those values
        return counter_seconds

def distribution_metrics(distribution_frequency, dataframe, col_dataframe):
    sorted_values = sorted(distribution_frequency.items(), key=lambda x: x[1], reverse=True) 
    q1 = np.percentile(dataframe[col_dataframe], 25)
    closest_q1 = min(list(distribution_frequency.keys()), key=lambda x: abs(x - q1))
    q2 = np.percentile(dataframe[col_dataframe], 50)
    closest_q2 = min(list(distribution_frequency.keys()), key=lambda x: abs(x - q2))
    q3 = np.percentile(dataframe[col_dataframe], 75)
    closest_q3 = min(list(distribution_frequency.keys()), key=lambda x: abs(x - q3))
    return [(sorted_values[0][0], sorted_values[0][1]),  # moda
            (closest_q1, distribution_frequency[closest_q1]),
            (closest_q2, distribution_frequency[closest_q2]),
            (closest_q3, distribution_frequency[closest_q3])]

def edgeset_dataframe_simple(transaction_dataframe, net):
    edgeset = set()
    for e in transaction_dataframe.iterrows():
        _edge_ = tuple([e[1].source, e[1].target, e[1].weight, e[1].epoch])
        edgeset.add(_edge_)
    edgeset_df = pd.DataFrame(edgeset, columns=['source', 'target', 'weight', 'epoch'])
    edgeset_df['time'] = edgeset_df.time.astype(str)
    # merge edgeset (no repeating) with total edgelist dataframe
    edgelist_g = nx.to_pandas_edgelist(net, edge_key="ekey")
    return pd.merge(edgeset_df, edgelist_g, how='left', 
                                on=['source', 'target', 'weight', 'epoch'])


def recirculation_graph(operations, net, output_folder):
    operations_set = edgeset_dataframe_simple(operations, net)
    re_g = nx.MultiDiGraph()
    for e in operations_set.iterrows():
        re_g.add_edge(e[1].source, e[1].target, 
                          weight=e[1].weight,
                          epoch=e[1].epoch, 
                          #operation_prom=e[1].operation_prom,
                          #dyad_prom=e[1].dyad_prom, ekey=e[1].ekey, 
                          #edge_status=e[1].edge_status
                          )
    #re_g = upgrade_graph_node_info_simple(re_g, output_folder)
    return operations_set, re_g

#####################################################
############## SET 4: dag analysis ##################
##################################################### 

def write_results_dag_analysis_summary(results, dag_type, graph_name, output_folder):
    f = open(output_folder+"/dag_analysis_summary.txt", "a")
    for k in results.keys():
        f.write(graph_name)
        f.write(';')
        f.write(dag_type)
        f.write(';')
        f.write(str(k))
        f.write(';')
        for i in results[k]:
            f.write(str(i))
            f.write(';')
        f.write('\n')
    f.close()
    
def subgraph_dag(net, _dag_type_):
    edge_status_dict = dict(nx.get_edge_attributes(net, 'edge_status'))
    edge_status_dict = {(k[0], k[1]):v for k,v in edge_status_dict.items()}
    # edgelist for subgraph
    edgelist_subgraph = [k for k,v in edge_status_dict.items() if v==_dag_type_]
    # DiGraph
    dnet = conversion_to_digraph(net)
    dag_dsubgraph = nx.edge_subgraph(dnet, edgelist_subgraph)
    # MultiDiGraph
    dag_subgraph = nx.MultiDiGraph()
    for e in net.edges(data=True):
        if edge_status_dict[(e[0], e[1])] == _dag_type_:
            dag_subgraph.add_edge(e[0], e[1], 
                                  weight=e[2]['weight'],
                                  epoch=e[2]['epoch'], 
                                  edge_status=e[2]['edge_status'])
    return dag_dsubgraph, dag_subgraph

def subgraph_scc(net, _scc_type_):
    edge_status_dict = dict(nx.get_edge_attributes(net, 'edge_status'))
    edge_status_dict = {(k[0], k[1]):v for k,v in edge_status_dict.items()}
    # edgelist for subgraph
    edgelist_subgraph = [k for k,v in edge_status_dict.items() if v==_scc_type_ ]
    # DiGraph
    dnet = conversion_to_digraph(net)
    scc_dsubgraph = nx.edge_subgraph(dnet, edgelist_subgraph)
    # MultiDiGraph
    scc_subgraph = nx.MultiDiGraph()
    for e in net.edges(data=True):
        if edge_status_dict[(e[0], e[1])] == _scc_type_:
            scc_subgraph.add_edge(e[0], e[1], 
                                    weight=e[2]['weight'],
                                    epoch =e[2]['epoch'], 
                                    edge_status=e[2]['edge_status'])
    return scc_dsubgraph, scc_subgraph
    

def write_dag_triadic_census(output_folder, graph_name, dag_type, results):
    f = open(output_folder+'/dag_triadic_census_summary.txt', "a")
    f.write(graph_name)
    f.write(';')
    f.write(dag_type)
    f.write(';')
    for r in results:
        f.write(str(r))
        f.write(';')
    f.write('\n')
    f.close()