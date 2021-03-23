def epsilonFunction(cutList, epsilon=0.25):
    # increase distance of continuous values from exact 0 and 1
    for i in range(len(cutList)):
        if (cutList[i] > 1 - epsilon):
            cutList[i] = 1 - epsilon
        if (cutList[i] < epsilon):
            cutList[i] = epsilon
    return cutList
