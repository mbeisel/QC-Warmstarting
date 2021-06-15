from copy import deepcopy

def epsilonFunction(cutList, epsilon=0.25):
    cut = deepcopy(cutList)
    epsilon = 0 if epsilon < 0 else epsilon
    epsilon = 0.5 if epsilon > 0.5 else epsilon
    # increase distance of continuous values from exact 0 and 1
    for i in range(len(cut)):
        if (cut[i] > 1 - epsilon):
            cut[i] = 1 - epsilon
        if (cut[i] < epsilon):
            cut[i] = epsilon
    return cut

def takeFirst(elem):
    return elem[0]
def takeSecond(elem):
    return elem[1]
def takeThird(elem):
    return elem[2]
def parseSolution(sol):
    return [int(i) for i in sol]

def hammingDistance(s1, s2, allowInverse = False):
    distance = 0
    for i,char in enumerate(s1):
        if char != s2[i]:
            distance+=1
    if allowInverse and distance > len(s1)/2:
        return len(s1) - distance
    return distance