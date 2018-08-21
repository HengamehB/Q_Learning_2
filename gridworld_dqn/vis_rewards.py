import matplotlib.pyplot as plt
import pickle
import sys

if __name__ == "__main__":
    pkl_filename = sys.argv[1]
    with open(pkl_filename, 'rb') as f:
        rewards = pickle.load(f)

    plt.plot(rewards, 'ro')
    plt.show()
