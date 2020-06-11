import argparse
from actor_critic.a2c.train import A2CTrainer
from params import Params
from actor_critic.inference import actor_critic_inference
from dqn.inference import dqn_inference
from actor_critic.evaluate import evaluate_actor_critic
from dqn.evaluate import evaluate_dqn

import numpy as np
import matplotlib.pyplot as plt


#  Performance plots
def plot_performance(scores):
    # Plot the policy performance
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(1, len(scores) + 1)
    y = scores
    plt.scatter(x, y, marker='x', c=y)
    fit = np.polyfit(x, y, deg=4)
    p = np.poly1d(fit)
    plt.plot(x, p(x), "r--")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Performance on CarRacing-v0')
    plt.show()



def get_trainer(model_type, params):
    model_path = 'models/' + model_type + '.pt'
    if model_type == 'a2c':
    return None


def run_training(model_type):
    params = Params('params/' + model_type + '.json')
    trainer = get_trainer(model_type, params)
    trainer.run()
    #plot_performance(scores)


def run_inference(model_type):
    params = Params('params/' + model_type + '.json')
    
    score, scores = actor_critic_inference(params, 'models/' + model_type + '.pt')

    print('Total score: {0:.2f}'.format(score))
    plot_performance(scores)


def run_evaluation(model_type):
    params = Params('params/' + model_type + '.json')
    
    score, scores = evaluate_actor_critic(params, 'models/' + model_type + '.pt')

    print('Average reward after 100 episodes: {0:.2f}'.format(score))
    plot_performance(scores)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True,
                        choices=['a2c'],
                        help='Which model to run / train.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true',
                       help='Train model.')
    group.add_argument('--inference', action='store_true',
                       help='Model inference.')
    group.add_argument('--evaluate', action='store_true',
                       help='Evaluate model on 100 episodes.')

    args = vars(parser.parse_args())
    if args['train']:
        run_training(args['model'])
    elif args['evaluate']:
        run_evaluation(args['model'])
    else:
        run_inference(args['model'])
