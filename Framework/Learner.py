'''
Comprehensive Framework for data preprocessing, model training, fine-tune, performance evaluation, visualization.
'''

from config import config_SL, config_meta
from Framework import IOManager, Visualizer, DataManager, ModelManager


class Learner():
    def __init__(self, config):
        self.config = config
        self.IOManager = IOManager.IOManager(self)
        self.visualizer = Visualizer.Visualizer(self)
        self.dataManager = DataManager.DataManager(self)
        self.modelManager = ModelManager.ModelManager(self)

    def setIO(self, type='train'):
        self.IOManager.initialize(type)
        self.IOManager.log.Info('Set IO Over.')

    def setVisualization(self):
        self.visualizer.initialize()
        self.IOManager.log.Info('Set Visualization Over.')

    def load_data(self):
        self.dataManager.load_data()
        self.IOManager.log.Info('Load Data Over.')

    def init_model(self):
        self.modelManager.init_model()
        self.IOManager.log.Info('Init Model Over.')

    def load_params(self):
        self.modelManager.load_params()
        self.IOManager.log.Info('Load Parameters Over.')

    def adjust_model(self):
        self.modelManager.adjust_model()
        self.IOManager.log.Info('Adjust Model Over.')

    def init_optimizer(self):
        self.modelManager.init_optimizer()
        self.IOManager.log.Info('Init Optimizer Over.')

    def def_loss_func(self):
        self.modelManager.def_loss_func()
        self.IOManager.log.Info('Define Loss Function Over.')

    def train_model(self):
        self.IOManager.log.Info('Train Model Start.')
        self.IOManager.log.Info('Learn Name: {}'.format(self.config.learn_name))
        print('=' * 200)
        self.IOManager.log.Info('Config: {}'.format(self.config))
        print('=' * 200)
        self.modelManager.train()
        self.visualizer.draw_train_test_curve()
        self.IOManager.log.Info('Train Model Over.')

    def test_model(self):
        self.IOManager.log.Info('Test Model Start.')
        self.IOManager.log.Info('Config: {}'.format(self.config))
        print('=' * 300)
        self.modelManager.test()
        self.IOManager.log.Info('Test Model Over.')

    def inference(self, type=None):
        self.IOManager.log.Info('Inference Start.')
        self.IOManager.log.Info('Config: {}'.format(self.config))
        print('=' * 300)
        self.modelManager.inference(type)
        self.IOManager.log.Info('Inference Over.')

    def reset_IOManager(self):
        self.IOManager = IOManager.IOManager(self)
        self.IOManager.initialize()
        self.IOManager.log.Info('Reset IOManager Over.')

    def reset_visualizer(self):
        self.visualizer = Visualizer.Visualizer(self)
        self.visualizer.initialize()
        self.IOManager.log.Info('Reset Visualizer Over.')

    def reset_dataManager(self):
        self.dataManager = DataManager.DataManager(self)
        self.dataManager.load_data()
        self.IOManager.log.Info('Reset DataManager Over.')

    def resset_modelManager(self):
        self.modelManager = ModelManager.ModelManager(self)
        self.modelManager.init_model()
        self.IOManager.log.Info('Reset ModelManager Over.')


if __name__ == '__main__':
    # config = config_SL.get_config()
    config = config_meta.get_config()
    learner = Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.test_model()
    learner.train_model()
    learner.test_model()

    # for i in range(1, 5):
    #     config.learn_name = 'train_{}'.format(i)
    #     learner.reset_IOManager()
    #     learner.reset_visualizer()
    #     learner.resset_modelManager()
    #     learner.init_optimizer()
    #     learner.def_loss_func()
    #     learner.train_model()
