import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Calculate weights')

parser.add_argument("--beta",
                    default=0.9,
                    type=float,
                    )
parser.add_argument("--nbr-bkg",
                    default=20,
                    )
args = parser.parse_args()


def calculate(beta, nbr_bkg):
    nbr_classes = np.array([nbr_bkg, 2.99, 0.37, 0.73])
    weights = (1-beta)/(1-beta**nbr_classes)
    print('weights', weights)


if __name__ == "__main__":

    calculate(args.beta, args.nbr_bkg)
