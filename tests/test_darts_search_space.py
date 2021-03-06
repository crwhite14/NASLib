import unittest
import logging
import torch
import os

from naslib.search_spaces import SimpleCellSearchSpace, DartsSearchSpace, HierarchicalSearchSpace, NasBench201SeachSpace
from naslib.optimizers import DARTSOptimizer, GDASOptimizer
from naslib.utils import utils, setup_logger

logger = setup_logger(os.path.join(utils.get_project_root().parent, "tmp", "tests.log"))
logger.handlers[0].setLevel(logging.FATAL)

config = utils.AttrDict()
config.dataset = 'cifar10'
config.search = utils.AttrDict()
config.search.grad_clip = None
config.search.learning_rate = 0.01
config.search.momentum = 0.1
config.search.weight_decay = 0.1
config.search.arch_learning_rate = 0.01
config.search.arch_weight_decay = 0.1
config.search.tau_max = 10
config.search.tau_min = 1
config.search.epochs = 2


data_train = (torch.ones([2, 3, 32, 32]), torch.ones([2]).long())
data_val = (torch.ones([2, 3, 32, 32]), torch.ones([2]).long())

if torch.cuda.is_available():
    data_train = tuple(x.cuda() for x in data_train)
    data_val = tuple(x.cuda() for x in data_val)


class DartsDartsIntegrationTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = DARTSOptimizer(config)
        self.optimizer.adapt_search_space(DartsSearchSpace())
        self.optimizer.before_training()
        

    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.8412, places=3)
        self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.8638, places=3)


    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        logits = final_arch(data_train[0])
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().cpu().numpy(), 0.0224, places=3)


class DartsGdasIntegrationTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = GDASOptimizer(config)
        self.optimizer.adapt_search_space(DartsSearchSpace())
        self.optimizer.before_training()


    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 1.7674, places=3)
        self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 5.7394, places=3)
    

    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        logits = final_arch(data_train[0])
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().cpu().numpy(), 0.0225, places=3)

if __name__ == '__main__':
    unittest.main()
