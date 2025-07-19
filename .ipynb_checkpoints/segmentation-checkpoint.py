from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as tree_to_petri
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.statistics.traces.generic.log import case_statistics

def split_event_log_by_cluster(trace_df, full_event_df, case_id='case:concept:name'):
    cluster_logs = {}
    for cluster_id in trace_df['cluster'].unique():
        case_ids = trace_df[trace_df['cluster'] == cluster_id][case_id]
        cluster_df = full_event_df[full_event_df[case_id].isin(case_ids)]

        cluster_df = dataframe_utils.convert_timestamp_columns_in_df(cluster_df)
        cluster_logs[cluster_id] = cluster_df

    return cluster_logs


def discover_process_models(cluster_logs, visualize=True):
    cluster_models = {}
    for cluster_id, cluster_df in cluster_logs.items():
        log = log_converter.apply(cluster_df, variant=log_converter.Variants.TO_EVENT_LOG)

        tree = inductive_miner.apply(log)
        net, im, fm = tree_to_petri.apply(tree)
        cluster_models[cluster_id] = (net, im, fm)

        if visualize:
            gviz = pn_visualizer.apply(net, im, fm)
            pn_visualizer.view(gviz)

    return cluster_models

def discover_process_models_by_cluster(trace_df, full_event_df, case_id='case:concept:name'):
    cluster_models = {}
    for cluster_id in trace_df['cluster'].unique():
        
        case_ids = trace_df[trace_df['cluster'] == cluster_id][case_id]
        cluster_df = full_event_df[full_event_df[case_id].isin(case_ids)]

        cluster_df = dataframe_utils.convert_timestamp_columns_in_df(cluster_df)
        log = log_converter.apply(cluster_df, variant=log_converter.Variants.TO_EVENT_LOG)

        tree = inductive_miner.apply(log)
        net, im, fm = tree_to_petri.apply(tree)
        cluster_models[cluster_id] = (net, im, fm)

        gviz = pn_visualizer.apply(net, im, fm);
        pn_visualizer.view(gviz)

    return cluster_models

def evaluate_models_cluster(models_dict, trace_df, full_event_df, case_col='case:concept:name', act_col='concept:name', time_col='time:timestamp'):
    results = []

    for cluster_id, (net, im, fm) in models_dict.items():
        case_ids = trace_df[trace_df['cluster'] == cluster_id][case_col]
        sublog = full_event_df[full_event_df[case_col].isin(case_ids)].copy()

        if sublog.empty:
            continue

        sublog = dataframe_utils.convert_timestamp_columns_in_df(sublog)
        event_log = log_converter.apply(sublog)

        num_places = len(net.places)
        num_transitions = len(net.transitions)
        num_nodes = num_places + num_transitions
        num_arcs = len(net.arcs)

        cnc = num_arcs / num_nodes if num_nodes else 0
        pt_cd = 0.5 * (num_arcs / num_places + num_arcs / num_transitions) if num_places and num_transitions else 0
        cyclomatic_number = num_arcs - num_nodes + 1

        connector_degrees = [len(p.in_arcs) + len(p.out_arcs) for p in net.places] + \
                            [len(t.in_arcs) + len(t.out_arcs) for t in net.transitions]
        acd = sum(connector_degrees) / len(connector_degrees) if connector_degrees else 0

        density = num_arcs / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        cfc = sum(len(t.out_arcs) for t in net.transitions if len(t.out_arcs) > 1)

        variants = case_statistics.get_variant_statistics(event_log)
        num_variants = len(variants)
        event_classes = set(event[act_col] for trace in event_log for event in trace)
        num_event_classes = len(event_classes)

        results.append({
            "Cluster": cluster_id,
            "# Nodes": num_nodes,
            "# Arcs": num_arcs,
            "CNC": cnc,
            "P/T-CD": pt_cd,
            "Cyclomatic Number (CN)": cyclomatic_number,
            "ACD": acd,
            "Density": density,
            "CFC": cfc,
            "# Event Classes": num_event_classes,
            "# Variants": num_variants,
            "# Events": len(sublog)
        })

    return pd.DataFrame(results)

def define_age_hierarchy(df):
    df['age_group'] = pd.cut(df['Age'], 
                             bins=[0, 7, 13, 19, 30, 40, 50, 60, 70, 80, 150], 
                             labels=['0-6', '7-12', '13-18', '19-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
                             right=False)
    df['age_category'] = df['age_group'].map({
        '0-6': 'Child',
        '7-12': 'Child',
        '13-18': 'Teen',
        '19-29': 'Young Adult',
        '30-39': 'Adult',
        '40-49': 'Adult',
        '50-59': 'Middle-aged',
        '60-69': 'Middle-aged',
        '70-79': 'Senior',
        '80+': 'Senior'
    })
    return df

def define_time_hierarchy(df):
    df['timestamp'] = pd.to_datetime(df['time:timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    return df

def define_activity_groups(df):
    activity_group_map = {
        'Admission IC': 'Admission',
        'Admission NC': 'Admission',
        'ER Registration': 'Emergency',
        'ER Triage': 'Emergency',
        'ER Sepsis Triage': 'Emergency',
        'Return ER': 'Emergency',
        'Leucocytes': 'Test',
        'CRP': 'Test',
        'LacticAcid': 'Test',
        'IV Antibiotics': 'Treatment',
        'IV Liquid': 'Treatment',
        'Release A': 'Release',
        'Release B': 'Release',
        'Release C': 'Release',
        'Release D': 'Release',
        'Release E': 'Release'
    }

    df['activity_group'] = df['concept:name'].map(activity_group_map)

    return df

def fill_trace_level_values(df, columns, case_col='case:concept:name'):
    df = df.copy()

    df[columns] = df.groupby(case_col)[columns].transform(lambda group: group.ffill().bfill())
    return df

def build_pcs():
    pcs = {
        'time': {
            'attributes': ['day', 'month', 'year'],
            'hierarchy': [('day', 'month'), ('month', 'year')]
        },
        'age': {
            'attributes': ['Age', 'age_group', 'age_category'],
            'hierarchy': [('Age', 'age_group'), ('age_group', 'age_category')]
        },
        'activity' : {
            'attributes': ['concept:name', 'activity_group'],
            'herarchy': [('concept:name', 'activity_group')]
        }
    }
    return pcs

def build_pcv(granularity: dict, filters: dict = {}):
    pcv = {
        'visible_dimensions': list(granularity.keys()),
        'granularity': granularity,
        'selection': filters
    }
    return pcv

def slice_pcv(pcv, dimension_to_remove, filter_values):
    new_pcv = copy.deepcopy(pcv)
    if dimension_to_remove in new_pcv['visible_dimensions']:
        new_pcv['visible_dimensions'].remove(dimension_to_remove)
    for attr, values in filter_values.items():
        new_pcv['selection'][attr] = values
    return new_pcv

def dice_pcv(pcv, filter_values):
    new_pcv = copy.deepcopy(pcv)
    for attr, values in filter_values.items():
        new_pcv['selection'][attr] = values
    return new_pcv

def rollup_pcv(pcv, pcs, dimension):
    new_pcv = copy.deepcopy(pcv)
    current_attr = new_pcv['granularity'].get(dimension)
    if not current_attr:
        return pcv

    hierarchy = pcs[dimension]['hierarchy']
    for low, high in hierarchy:
        if low == current_attr:
            new_pcv['granularity'][dimension] = high
            return new_pcv
    return pcv 

def drilldown_pcv(pcv, pcs, dimension):
    new_pcv = copy.deepcopy(pcv)
    current_attr = new_pcv['granularity'].get(dimension)
    if not current_attr:
        return pcv

    hierarchy = pcs[dimension]['hierarchy']
    for low, high in hierarchy:
        if high == current_attr:
            new_pcv['granularity'][dimension] = low
            return new_pcv
    return pcv 

def materialize_process_cube_view(df, pcs, pcv):
    filtered_df = df.copy()
    for attr, values in pcv['selection'].items():
        filtered_df = filtered_df[filtered_df[attr].isin(values)]
    
    gran_attrs = [pcv['granularity'][dim] for dim in pcv['visible_dimensions']]

    for attr in gran_attrs:
        if filtered_df[attr].dtype.name == 'category':
            filtered_df[attr] = filtered_df[attr].cat.add_categories('Unknown').fillna('Unknown')
        elif pd.api.types.is_numeric_dtype(filtered_df[attr]):
            filtered_df[attr] = filtered_df[attr].fillna(-1) 
        else:
            filtered_df[attr] = filtered_df[attr].fillna('Unknown')

    cube = {}
    grouped = filtered_df.groupby(gran_attrs)

    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,) 
        cube[key] = group.copy()

    return cube # dict (dimensions: sublog)

def discover_models_for_cells(df, cell_list, case_col='case:concept:name', act_col='concept:name', time_col='time:timestamp'):
    models = {}

    for cell in cell_list:
        year, age_cat, act_group = cell
        sublog = df[
            (df['year'] == year) &
            (df['age_category'] == age_cat) &
            (df['activity_group'] == act_group)
        ]

        if sublog.empty:
            print(f"Cell {cell}: Skip, since no log")
            continue

        print(f"Cell {cell} Number of events: {len(sublog)}")

        sublog = dataframe_utils.convert_timestamp_columns_in_df(sublog)
        event_log = log_converter.apply(sublog, variant=log_converter.Variants.TO_EVENT_LOG)
        tree = inductive_miner.apply(event_log)

        net, im, fm = tree_to_petri.apply(tree)

        gviz = pn_visualizer.apply(net, im, fm)
        pn_visualizer.view(gviz)  

        models[cell] = (net, im, fm)

    return models

def evaluate_models_pc(df, models_dict, case_col='case:concept:name', act_col='concept:name', time_col='time:timestamp'):
    results = []

    for cell_key, (net, im, fm) in models_dict.items():
        year, age_cat, act_group = cell_key

        sublog = df[
            (df['year'] == year) &
            (df['age_category'] == age_cat) &
            (df['activity_group'] == act_group)
        ].copy()

        if sublog.empty:
            continue

        sublog = dataframe_utils.convert_timestamp_columns_in_df(sublog)
        event_log = log_converter.apply(sublog)

        num_places = len(net.places)
        num_transitions = len(net.transitions)
        num_nodes = num_places + num_transitions
        num_arcs = len(net.arcs)

        cnc = num_arcs / num_nodes if num_nodes else 0
        pt_cd = 0.5 * (num_arcs / num_places + num_arcs / num_transitions) if num_places and num_transitions else 0
        cyclomatic_number = num_arcs - num_nodes + 1

        connector_degrees = [len(p.in_arcs) + len(p.out_arcs) for p in net.places] + \
                            [len(t.in_arcs) + len(t.out_arcs) for t in net.transitions]
        acd = sum(connector_degrees) / len(connector_degrees) if connector_degrees else 0

        density = num_arcs / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        cfc = sum(len(t.out_arcs) for t in net.transitions if len(t.out_arcs) > 1)

        variants = case_statistics.get_variant_statistics(event_log)
        num_variants = len(variants)
        event_classes = set(event["concept:name"] for trace in event_log for event in trace)
        num_event_classes = len(event_classes)

        results.append({
            "Cell": cell_key,
            "# Nodes": num_nodes,
            "# Arcs": num_arcs,
            # High CNC means increase of complexity (cycles are more challenging to understand than sequential ones)
            "CNC": cnc, # Arcs/Nodes
            "P/T-CD": pt_cd, # Density of arcs between transitions and places
            "Cyclomatic Number (CN)": cyclomatic_number, # the number of linearly independent paths in a process model (No repetitions in the path)
            "ACD": acd, # Average of number of arcs in/out from the connectors
            "Density": density, # actual number of arcs/possible number of arcs
            "CFC": cfc,# Aggregate of out-degree of splits (e.g. OR, XOR, AND)
            "# Event Classes": num_event_classes,
            "# Variants": num_variants,
            "# Events": len(sublog)
        })

    return pd.DataFrame(results)

## Differences in terms of output
## Comparison of the output
# at 3 on tuesday
