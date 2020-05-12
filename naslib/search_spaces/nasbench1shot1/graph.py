from torch import nn

from naslib.optimizers.core.optimizer import NASOptimizer
from naslib.search_spaces.core import NodeOpGraph
from naslib.search_spaces.nasbench1shot1 import PRIMITIVES
from naslib.search_spaces.nasbench1shot1.primitives import Stem, OPS
from naslib.utils import config_parser


class Cell(NodeOpGraph):
    def __init__(self, primitives, cell_type, C_prev, C, reduction_prev, ops_dict, num_intermediate_nodes, *args,
                 **kwargs):
        self.primitives = primitives
        self.cell_type = cell_type
        self.C_prev = C_prev
        self.C = C
        self.reduction_prev = reduction_prev
        self.ops_dict = ops_dict
        self.num_intermediate_nodes = num_intermediate_nodes
        self.drop_path_prob = 0
        super(Cell, self).__init__(*args, **kwargs)

    def _build_graph(self):
        # Input Nodes: Previous / Previous-Previous cell

        self.add_node(0, type='input', desc='previous')

        # 4 intermediate nodes
        for i in range(self.num_intermediate_nodes):
            self.add_node(len(self.nodes), type='inter', comb_op='sum')

        # Output node
        self.add_node(len(self.nodes), type='output', comb_op='cat_channels')

        # Edges: input-inter and inter-inter
        for to_node in self.inter_nodes():
            for from_node in range(to_node):
                stride = 2 if self.cell_type == 'reduction' and from_node < 2 else 1
                self.add_edge(
                    from_node, to_node, op=None, op_choices=self.primitives,
                    op_kwargs={'C': self.C, 'stride': stride, 'out_node_op': 'sum', 'ops_dict': self.ops_dict,
                               'affine': False}, to_node=to_node, from_node=from_node)

        # Edges: inter-output
        self.add_edge(2, 6, op=Identity())
        self.add_edge(3, 6, op=Identity())
        self.add_edge(4, 6, op=Identity())
        self.add_edge(5, 6, op=Identity())


class MacroGraph(NodeOpGraph):
    def __init__(self, config, primitives, ops_dict, *args, **kwargs):
        self.config = config
        self.primitives = primitives
        self.ops_dict = ops_dict
        super(MacroGraph, self).__init__(*args, **kwargs)

    def _build_graph(self):
        num_layers = self.config['layers']
        C = self.config['init_channels']
        C_curr = C
        stem = Stem(C_curr)
        C_prev = C

        self.add_node(0, type='input')
        self.add_node(1, op=stem, type='stem')

        # Normal and pooling layers
        cell_num = 0
        for layer_num in range(num_layers):
            if layer_num in [num_layers // 3, 2 * num_layers // 3]:
                self.add_node(cell_num + 2,
                              op=nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
                C_curr *= self.config['channel_multiplier']
                cell_num += 1
            self.add_node(cell_num + 2,
                          op=nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                          type='normal')
            cell_num += 1
            C_prev = C_curr

        pooling = nn.AdaptiveAvgPool2d(1)
        classifier = nn.Linear(C_prev, self.config['num_classes'])

        self.add_node(cell_num + 2, op=pooling, transform=lambda x: x[0], type='pooling')
        self.add_node(cell_num + 3, op=classifier, transform=lambda x: x[0].view(x[0].size(0), -1), type='output')

        # Edges
        for i in range(1, cell_num + 4):
            self.add_edge(i - 1, i, type='input', desc='previous')

        # From output of normal-reduction cell to pooling layer
        self.add_edge(cell_num + 1, cell_num + 2)
        self.add_edge(cell_num + 2, cell_num + 3)
        pass

    def get_cells(self, cell_type):
        cells = list()
        for n in self.nodes:
            if 'type' in self.nodes[n] and self.nodes[n]['type'] == cell_type:
                cells.append(n)
        return cells


if __name__ == '__main__':
    config = config_parser('../../configs/nasbench_1shot1.yaml')

    one_shot_optimizer = NASOptimizer.from_config(**config)
    search_space = MacroGraph.from_optimizer_op(
        one_shot_optimizer,
        config=config,
        primitives=PRIMITIVES,
        ops_dict=OPS
    )
    one_shot_optimizer.init()
