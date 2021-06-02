def epsilonFunction(cutList, epsilon=0.25):
    # increase distance of continuous values from exact 0 and 1
    for i in range(len(cutList)):
        if (cutList[i] > 1 - epsilon):
            cutList[i] = 1 - epsilon
        if (cutList[i] < epsilon):
            cutList[i] = epsilon
    return cutList

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


