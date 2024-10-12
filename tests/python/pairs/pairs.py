#!/bin/python3
import numpy as np
import os
import argparse

def main(dataset, epsilon):
    data = np.genfromtxt(dataset, delimiter=',', dtype=np.float64, usemask=True)
    print(f"data.shape: {data.shape}")
    print(f"epsilon: {epsilon}")
    

    # Compute euclidean distance pairs in double precision
    # We don't have enough memory to do it on one go, so break it down
    pointsPerIter = 1000
    # Calculate sum of squares once
    sums = np.sum(data ** 2, axis=1)
    numPairs = 0;
    indices = np.ndarray((0, 2))
    for firstIndex in range(0, data.shape[0], pointsPerIter):
    #for firstIndex in range(0, 1000, pointsPerIter):
        endIndex = min(data.shape[0], firstIndex + pointsPerIter)
        print(f"endIndex: {endIndex}")
        
        dataSubset = data[firstIndex:endIndex]

        dists = sums.T + np.dot(-2 * dataSubset, data.T) + sums[firstIndex:endIndex].reshape(-1, 1)
        dists = np.sqrt(abs(dists))
        
        pairs = dists < epsilon
        subsetNumPairs = np.sum(pairs)
        print(f"subsetNumPairs: {subsetNumPairs}")
        
        numPairs += subsetNumPairs
        # Save out all pairs that we found. Assume epsilon is large, so the resulting file won't be too crazy
        localIndices = np.transpose(np.nonzero(pairs))
        # Have to add in the base index since we are calculating relative indices
        localIndices[:, 0] += firstIndex
        indices = np.vstack((indices, localIndices))

    pairsFileName = f'pairs/{os.path.basename(dataset.name)}_pairs_eps_{epsilon}.out'
    print(f"pairsFileName: {pairsFileName}")
    
    np.savetxt(pairsFileName, indices, delimiter=',', fmt='%u')

    print(f"numPairs: {numPairs}")
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Vector Data Parser")
    parser.add_argument("dataset", help="A csv file containing the vector data you want to analyze", type=argparse.FileType('r'))
    parser.add_argument("epsilon", help="The Epsilon value to evaluate pairs over", type=float)
    args = parser.parse_args()
    main(args.dataset, args.epsilon)
