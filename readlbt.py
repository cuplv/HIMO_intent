import graph_tool.all as gt
# from __future__ import print_function

def _split2(x):
    x = x.strip()
    space = x.find(' ')
    newline = x.find('\n')
    if newline == -1:
        nexts = space
    elif space == -1:
        nexts = newline
    else:
        nexts = min(space, newline)
    if nexts == -1:
        return [x]
    first = x[:nexts]
    x = x[nexts:].strip()
    space = x.find(' ')
    newline = x.find('\n')
    if newline == -1:
        nexts = space
    elif space == -1:
        nexts = newline
    else:
        nexts = min(space, newline)
    if nexts == -1:
        return [first, x]
    second = x[:nexts]
    return [first, second, x[nexts:]]


def convert_LBTT_to_graph(gbastr):
    """Construct automaton from scheck output.
    `gbastr` is a string that defines the generalized Büchi automaton.
    """
    parts = _split2(gbastr)
    state_count = int(parts[0])

    # A = GBAutomaton(int(parts[1]))
    A = gt.Graph()
    vertex = dict()
    edges = dict()
    v_accept = A.new_vertex_property("object")
    v_init = A.new_vertex_property("bool")
    e_gate = A.new_edge_property("string")

    state_parts = [part for part in parts[2].split('-1')
                   if len(part.strip()) > 0]

    for (ii, state_part) in enumerate(state_parts):
        if ii % 2 == 0:
            x = _split2(state_part)
            state_name = int(x[0])
            initial = True if x[1] == '1' else False
            if len(x) > 2:
                acceptance_sets=[int(ac) for ac in x[2].split()]
            else:
                acceptance_sets = []

            try:
                A.vertex(state_name).is_valid()
            except:
                vertex[state_name] = A.add_vertex()
            v_accept[vertex[state_name]] = acceptance_sets
            v_init[vertex[state_name]] = initial

        else:  # Transitions
            succ = None
            next_succ = None
            gate = ''
            for transitions_part in state_part.split():
                try:
                    next_succ = int(transitions_part)
                except ValueError:
                    next_succ = None
                if succ is None and next_succ is not None:
                    succ = next_succ
                    next_succ = None
                    continue
                if next_succ is not None:
                    # A.add_edge(state_name, succ, gate=gate)
                    try:
                        A.vertex(state_name).is_valid()
                    except:
                        vertex[state_name] = A.add_vertex()
                    try:
                        A.vertex(succ).is_valid()
                    except:
                        vertex[succ] = A.add_vertex()
                    edges[vertex[state_name], vertex[succ]] = A.add_edge(vertex[state_name], vertex[succ])
                    e_gate[edges[vertex[state_name], vertex[succ]]] = gate

                    succ = next_succ
                    next_succ = None
                    gate = ''
                else:
                    if len(gate) > 0:
                        gate += ' '
                    gate += transitions_part
            if succ is not None:
                # A.add_edge(state_name, succ, gate=gate)
                try:
                    A.vertex(state_name).is_valid()
                except:
                    vertex[state_name] = A.add_vertex()
                try:
                    A.vertex(succ).is_valid()
                except:
                    vertex[succ] = A.add_vertex()
                edges[vertex[state_name], vertex[succ]] = A.add_edge(vertex[state_name], vertex[succ])
                e_gate[edges[vertex[state_name], vertex[succ]]] = gate

    return A, vertex, edges, v_accept, v_init, e_gate

# def generate_Buchi(spot_result):

#     (A, vertex, edges, v_accept, v_init, e_gate) = readlbt(spot_result)
#     # print("Type: ", type(gba), "\n")
#     # print("Nodes: ", gba.nodes, "\n")
#     # print("Nodes.data: ", gba.nodes.data(), "\n")
#     # print("Edges: ",gba.edges, "\n")
#     # print("Edges.data ",gba.edges.data())
#     return A, vertex, edges, v_accept, v_init, e_gate