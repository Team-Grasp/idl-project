import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--num_episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=3e-4, help="The learning rate.")

    parser.add_argument('--model_path', dest='model_path', type=str,
                    default="", help="Existing model to use.")

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