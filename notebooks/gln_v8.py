from numpy import *
from collections import Counter

def rotate_y(pts, theta):
    R = array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
    return dot(R, pts.T).T

def rotate_x(pts, theta):
    R = array([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])
    return dot(R, pts.T).T

def rotate_z(pts, theta):
    R = array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
    return dot(R, pts.T).T

class simplify_v8:
    def __init__(self, adj_list, net):
        self.adj_list = adj_list
        self.net = net.copy()
        self.edges = net['links'].copy()
        
        a = list(self.edges.keys())
        
        for edge in a:
            if self.edges[edge]['end_points'][0] <= self.edges[edge]['end_points'][1]:
                if '[%s, %s]' % (self.edges[edge]['end_points'][0], self.edges[edge]['end_points'][1]) not in self.edges:
                    self.edges['[%s, %s]' % (self.edges[edge]['end_points'][0], self.edges[edge]['end_points'][1])] = {0: self.edges[edge]}
                else:
                    self.edges['[%s, %s]' % (self.edges[edge]['end_points'][0], self.edges[edge]['end_points'][1])][len(self.edges['[%s, %s]' % (self.edges[edge]['end_points'][0], self.edges[edge]['end_points'][1])])] = self.edges[edge]
                del self.edges[edge]
            else:
                new_end_points = self.edges[edge]['end_points'][::-1]
                if '[%s, %s]' % (new_end_points[0], new_end_points[1]) not in self.edges:
                    self.edges['[%s, %s]' % (new_end_points[0], new_end_points[1])] = {0: {'end_points': new_end_points, 'points': self.edges[edge]['points'][::-1], 'radius': self.edges[edge]['radius']}}
                else:
                    self.edges['[%s, %s]' % (new_end_points[0], new_end_points[1])][len(self.edges['[%s, %s]' % (new_end_points[0], new_end_points[1])])] = {'end_points': new_end_points, 'points': self.edges[edge]['points'][::-1], 'radius': self.edges[edge]['radius']}
                del self.edges[edge]
            

    def input_base_loops(self, loops):
        self.base_loop = {}
        m = len(loops)
        for i in range(m):
            self.base_loop[i] = {'loop': loops[i]}
            points = []
            for j in range(len(loops[i]) - 1):
                if loops[i][j] < loops[i][j + 1]:
                    points += self.edges['[%s, %s]' % (str(loops[i][j]), str(loops[i][j + 1]))][0]['points'][1:]
                else:
                    points += self.edges['[%s, %s]' % (str(loops[i][j + 1]), str(loops[i][j]))][0]['points'][::-1][1:]
            points.append(points[0])
            self.base_loop[i]['points'] = points
            
    def input_base_loops_2(self, loops):
        self.base_loop = loops
        
    def find_base_loops(self, num_source):  # notice that points does not include same end node twice
        self.base_loop = {}
        loop_set = set()
        
        ### dealing with multi-edges ###
        for edge in self.edges.values():
            if len(edge) > 1:  # multi-edge
                loop_set.add(frozenset(edge[0]['end_points']))
                for i in range(len(edge)):  # for each pair of multi_edges
                    for j in range(i + 1, len(edge)):
                        seg_1 = edge[i]['points']
                        seg_2 = edge[j]['points'][::-1]
                        loop = seg_1 + seg_2[1:]  # the starting and ending points are the same
                        self.base_loop[len(self.base_loop)] = {'loop': edge[0]['end_points'] + [edge[0]['end_points'][0]], 'points': loop}
                        self.base_loop[len(self.base_loop) - 1]['nei'] = {edge[0]['end_points'][0]: [edge[0]['end_points'][1], edge[0]['end_points'][1]], edge[0]['end_points'][1]: [edge[0]['end_points'][0], edge[0]['end_points'][0]]}
        
        N = len(self.net['nodes']['positions'])
        for n in range(num_source):
            gnodes = set(self.net['nodes']['labels'])
            root = None
            while gnodes:  # loop over connected components
                if root is None:
                    a = list(gnodes)
                    if n < len(a):
                        tmpn = int(N * random.rand())
                        a = a[tmpn:] + a[:tmpn]
                        root = a[0]
                    else:
                        root = a[-1]
                    gnodes.remove(root)
                stack = [root]
                pred = {root: root}
                used = {root: set()}
                while stack:  # walk the spanning tree finding cycles
                    z = stack.pop()  # use last-in so cycles easier to find
                    zused = used[z]
                    for nbr in self.adj_list[z]:
                        if nbr not in used:  # new node
                            pred[nbr] = z
                            stack.append(nbr)
                            used[nbr] = set([z])
                        elif nbr == z:  # self loops
                            if set([z]) not in loop_set:
                                for i in self.edges['[%s, %s]' % (str(z), str(z))].keys():
                                    self.base_loop[len(self.base_loop)] = {'loop': [z, z], 'points': self.edges['[%s, %s]' % (str(z), str(z))][i]['points']}
                                loop_set.add(frozenset([z]))
                        elif nbr not in zused:  # found a cycle
                            pn = used[nbr]
                            cycle = [nbr, z]
                            p = pred[z]
                            while p not in pn:
                                cycle.append(p)
                                p = pred[p]
                            cycle.append(p)
                            if set(cycle) not in loop_set:
                                loop_set.add(frozenset(cycle))
                                self.base_loop[len(self.base_loop)] = {'loop': cycle}
                                points = []
                                cycle.append(cycle[0])
                                for i in range(len(cycle) - 1):
                                    if cycle[i] < cycle[i + 1]:
                                        points += self.edges['[%s, %s]' % (str(cycle[i]), str(cycle[i + 1]))][0]['points'][1:]
                                    else:
                                        points += self.edges['[%s, %s]' % (str(cycle[i + 1]), str(cycle[i]))][0]['points'][::-1][1:]
                                points = [points[-1]] + points
                                self.base_loop[len(self.base_loop) - 1]['points'] = points
                                self.base_loop[len(self.base_loop) - 1]['nei'] = {cycle[h]: [cycle[h - 1], cycle[h + 1]] for h in range(1, len(cycle) - 1)}
                                self.base_loop[len(self.base_loop) - 1]['nei'][cycle[0]] = [cycle[-2], cycle[1]]
                            used[nbr].add(z)
                gnodes -= set(pred)
                root = None
                
        self.loop_set = set(frozenset(self.base_loop[i]['loop']) for i in range(len(self.base_loop)))
    
    # Other methods (cal_linking_num, cal_m2_old, etc.) remain unchanged