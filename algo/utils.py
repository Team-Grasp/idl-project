import argparse
import collections

AvgMetricStore = collections.namedtuple(
    'AvgMetricStore', ['reward', 'success_rate',
                       'entropy_loss', 'pg_loss', 'value_loss', 'loss'])
RolloutResults = collections.namedtuple(
    'RolloutResults', ['gradients', 'parameters', 'metrics'])

class MetricStore(object):
    def __init__(self):
        self.total_reward = 0.0
        self.total_success_rate = 0.0
        self.total_entropy_loss = 0.0
        self.total_pg_loss = 0.0
        self.total_value_loss = 0.0
        self.total_loss = 0.0

    def add(self, metrics):
        reward, success_rate, entropy_loss, pg_loss, value_loss, loss = metrics
        self.total_reward += reward
        self.total_success_rate += success_rate
        self.total_entropy_loss += entropy_loss
        self.total_pg_loss += pg_loss
        self.total_value_loss += value_loss
        self.total_loss += loss

    def avg(self, count):
        count = float(count)
        return AvgMetricStore(self.total_reward / count,
                              self.total_success_rate / count,
                              self.total_entropy_loss / count,
                              self.total_pg_loss / count,
                              self.total_value_loss / count,
                              self.total_loss / count)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--num_episodes', dest='num_episodes', type=int,
                        default=10000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=3e-4, help="The learning rate.")
    parser.add_argument('--seed', dest='seed', type=int,
                        help="Random seed.")

    parser.add_argument('--algo_name', dest='algo_name', type=str,
                        default="", help="Type of Algorithm to use. Select from [MAML, REPTILE, REPTILIAN_MAML]")
    parser.add_argument('--model_path', dest='model_path', type=str,
                        default="", help="Existing model to use.")
    # parser.add_argument('--train_targets_path', dest='train_targets_path', type=str,
    #                     default="", help="Existing train_targets to use.")
    # parser.add_argument('--test_targets_path', dest='test_targets_path', type=str,
    #                     default="", help="Existing test_targets to use.")

    # Mode
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    parser.add_argument('--train', dest='train',
                        action='store_true',
                        help="Whether to train model.")
    parser.add_argument('--eval', dest='train',
                        action='store_false',
                        help="Whether to evaluate an existing model.")
    parser.set_defaults(train=False)

    return parser.parse_args()
