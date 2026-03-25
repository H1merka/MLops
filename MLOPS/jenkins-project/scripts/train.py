"""Wrapper script to call train_model.train().
Saves artifacts model_bundle.pkl and best_model.txt in repository root.
"""
from train_model import train


def main():
    print('Starting training...')
    train()


if __name__ == '__main__':
    main()
