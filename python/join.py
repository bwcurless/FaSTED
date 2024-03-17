#!/bin/python3
import numpy as np
import argparse

def main(dataset):
    data = np.genfromtxt(dataset, delimiter=',', usemask=True)
    print(f"data.shape: {data.shape}")

    
    






if __name__ == '__main__':
    parser = argparse.ArgumentParser("Vector Data Parser")
    parser.add_argument("dataset", help="A csv file containing the vector data you want to analyze", type=argparse.FileType('r'))
    args = parser.parse_args()
    main(args.dataset)
