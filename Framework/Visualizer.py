import seaborn as sns
import matplotlib.pyplot as plt


class Visualizer():
    def __init__(self, learner):
        self.learner = learner
        self.IOManager = learner.IOManager
        self.config = learner.config

    def initialize(self):
        self.step_log_interval = []
        self.train_metric_record = []
        self.train_loss_record = []
        self.step_valid_interval = []
        self.valid_metric_record = []
        self.valid_loss_record = []
        self.step_test_interval = []
        self.test_metric_record = []
        self.test_loss_record = []

    def draw_train_test_curve(self):
        sns.set(style="darkgrid")
        plt.figure(22, figsize=(16, 12))
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        plt.subplot(2, 2, 1)
        plt.title("Train Acc Curve", fontsize=23)
        plt.xlabel("Step", fontsize=20)
        plt.ylabel("Accuracy", fontsize=20)
        plt.plot(self.step_log_interval, self.train_metric_record)
        plt.subplot(2, 2, 2)
        plt.title("Train Loss Curve", fontsize=23)
        plt.xlabel("Step", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.plot(self.step_log_interval, self.train_loss_record)
        plt.subplot(2, 2, 3)
        plt.title("Test Acc Curve", fontsize=23)
        plt.xlabel("Epoch", fontsize=20)
        plt.ylabel("Accuracy", fontsize=20)
        plt.plot(self.step_test_interval, self.test_metric_record)
        plt.subplot(2, 2, 4)
        plt.title("Test Loss Curve", fontsize=23)
        plt.xlabel("Step", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.plot(self.step_test_interval, self.test_loss_record)

        plt.savefig('{}/{}.{}'.format(self.IOManager.result_path, self.config.learn_name, self.config.save_figure_type))
        plt.show()
