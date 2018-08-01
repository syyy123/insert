
#coding:'utf-8'

import pandas as pd
import numpy as np
from scipy import spatial
import copy
import datetime
import matplotlib.pyplot as plt
import cx_Oracle as oracle


'''递归查找坐标，解决坐标误差问题'''
def plotPipesA(xB, yB, diff):
    global p1
    global count
    for i in range(len(p1)):
        if xB - diff < p1[i, 1, 0] < xB + diff and yB - diff < p1[i, 2, 0] < yB + diff and  p1[i, 10, 0] == 0:
            p1[i, 10, 0] = 1
            p1[i, 1, 0], p1[i, 2, 0] = xB, yB
            plotPipesA(p1[i, 3, 0], p1[i, 4, 0], diff)

        if xB - diff < p1[i, 3, 0]  < xB + diff and yB - diff < p1[i, 4, 0] < yB + diff and  p1[i, 10, 0] == 0:
            p1[i, 10, 0] = 1
            p1[i, 3, 0], p1[i, 4, 0] = xB, yB
            plotPipesA(p1[i, 1, 0], p1[i, 2, 0], diff)


def setDeviceStatus(x, y, x0, y0, side, cp, side2):
    global p1
    global t1
    global tPLc
    global tPLm
    p1[cp, 17, 0] = 1

    for j in range(len(t1)):
        if x - diff < t1[j, 3] < x + diff and y - diff < t1[j, 4] < y + diff:
            t1[j, 6] = 2 if side2 == 1 else 1

    for k in range(len(p1)):
        x1, y1 = p1[k, 1, 0], p1[k, 2, 0]
        x2, y2 = p1[k, 3, 0], p1[k, 4, 0]

        if (x1 == x and y1 == y) and ((x2 != x0 or y2 != y0)) and p1[k, 17, 0] == 0 and checkside(x2,y2,tPLm,tPLc,side2) == 1:
            setDeviceStatus(x2, y2, x1, y1, side, k, side2)
        if (x2 == x and y2 == y) and ((x1 != x0 or y1 != y0)) and p1[k, 17, 0] == 0 and checkside(x1,y1,tPLm,tPLc,side2) == 1:
            setDeviceStatus(x1, y1, x2, y2, side, k, side2)


def find_connection():
    global conCount
    conCount = 0

    setDeviceStatus(-25800.188, 20249.748, -25756.757, 20285.65, 2, 92, 1)
    setDeviceStatus(-20159.2086, 11813.8433, -20152.4087, 11812.0661, 1, 198, 2)

    for i in range(len(p1)):
        p1[:, 17, 0] = 0
        length = p1[i, 8, 0]
        if checkside(p1[i, 1, 0], p1[i, 2, 0], tPLm, tPLc, 1) == 1:
            getConnected(p1[i, 1, 0], p1[i, 2, 0], p1[i, 3, 0], p1[i, 4, 0], 1, i, i, length, 1)
        p1[:, 17, 0] = 0
        length = p1[i, 8, 0]
        if checkside(p1[i, 3, 0], p1[i, 4, 0], tPLm, tPLc, 1) == 1:
            getConnected(p1[i, 3, 0], p1[i, 4, 0], p1[i, 1, 0], p1[i, 2, 0], 2, i, i, length, 1)
        p1[:, 17, 0] = 0

    for i in range(len(p1[:, 1, 0])):
        p1[:, 17, 0] = 0
        length = p1[i, 8, 0]
        if checkside(p1[i, 1, 0], p1[i, 2, 0], tPLm, tPLc, 2) == 1:
            getConnected(p1[i, 1, 0], p1[i, 2, 0], p1[i, 3, 0], p1[i, 4, 0], 1, i, i, length, 2)
        p1[:, 17, 0] = 0
        length = p1[i, 8, 0]
        if checkside(p1[i, 3, 0], p1[i, 4, 0], tPLm, tPLc, 2) == 1:
            getConnected(p1[i, 3, 0], p1[i, 4, 0], p1[i, 1, 0], p1[i, 2, 0], 2, i, i, length, 2)
        p1[:, 17, 0] = 0


def getConnected(x, y, x0, y0, side, i, cp, length, side2):
    global p1
    global t1
    global conCount
    global tPLc
    global tPLm
    p1[cp, 17, 0] = 1

    for j in range(len(t1)):
        if x - diff < t1[j, 3] < x + diff and y - diff < t1[j, 4] < y + diff and t1[j, 6] != side2:
            if side == 1:
                p1[i, 19, int(p1[i, 18, 0])] = t1[j, 0]
                p1[i, 22, int(p1[i, 18, 0])] = length
                p1[i, 18, 0] = p1[i, 18, 0] + 1

            elif side == 2:
                p1[i, 21, int(p1[i, 20, 0])] = t1[j, 0]
                p1[i, 23, int(p1[i, 20, 0])] = length
                p1[i, 20, 0] = p1[i, 20, 0] + 1


    for k in range(len(p1)):
        x1, y1 = p1[k, 1, 0], p1[k, 2, 0]
        x2, y2 = p1[k, 3, 0], p1[k, 4, 0]

        if (x1 == x and y1 == y) and ((x2!=x0 or y2!=y0)) and p1[k, 17, 0] == 0:
            l = length + p1[k, 8, 0]
            getConnected(x2, y2, x1, y1, side, i, k, l, side2)

        if (x2 == x and y2 == y) and ((x1!=x0 or y1!=y0)) and p1[k, 17, 0] == 0:
            l2 = length + p1[k, 8, 0]
            getConnected(x1, y1, x2, y2, side, i, k, l2, side2)


def getPL(x, y, x0, y0):
    m =(y[-1] - y[0]) / (x[-1] - x[0])
    PLm = np.tan(np.arctan(m) + np.pi / 2)
    PLc = y0 - PLm * x0
    plX = x
    plY = [PLm * i + PLc for i in plX]

    return plX, plY, PLm, PLc


def getFitMainLine(xIn, yIn):
    linearCoefficients = np.polyfit(xIn, yIn, 1)
    x = np.linspace(min(xIn), max(xIn), 50)
    y = np.polyval(linearCoefficients, x)

    return x, y


def fillSamePoints(x, y, p, m, c, side):
    global p1
    for i in range(len(p1)):
        if p1[i, 1, 0] == x and p1[i, 2, 0] == y:
            p1[i, 6, 0] = p
            if p1[i, 15, 0] == 0:
                p1[i, 15, 0] = 3
            if checkside(p1[i, 1, 0], p1[i, 2, 0], m, c, side) == 1:
                p1[i, 11, 0] = 1

        if p1[i, 3, 0] == x and p1[i, 4, 0] == y:
            p1[i, 7, 0] = p
            if p1[i, 16, 0] == 0:
                p1[i, 16, 0] = 3
            if checkside(p1[i, 3, 0], p1[i, 4, 0], m, c, side) == 1:
                p1[i, 12, 0] = 1


def checkside(x, y, m, c, angle):
    if angle == 1:
        isFront = 1 if y > m * x + c else 0
    else:
        isFront = 1 if y < m * x + c else 0

    return isFront


def isConnected(div, pip, s, m, c, ang, side, m2, c2, side2):
    global p1
    global t1
    status=0
    if p1[int(pip), 18, 0] + p1[int(pip), 20, 0] >= 50:
        temp=checkside(t1[div, 3], t1[div, 4], m, c, ang)
        temp2 = checkside(t1[div, 3], t1[div, 4], m2, c2, side2)
        if temp == side and temp2 == 1:
            status=2
    else:
        if s == 1:
            for i in range(int(p1[pip, 18, 0])):
                if p1[pip,19,i]==t1[div, 0]:
                    status = 1
                    break
        elif s==2:
            for i in range(int(p1[pip, 20, 0])):
                 if p1[pip, 21, i] == t1[div, 0]:
                    status = 1
                    break
    return status


def findPressureBasic(j, s, p, m, c, side2, gradiant):
    global t1
    global count3
    global p1
    if s == 2:
        x = p1[j, 3, 0]
        y = p1[j, 4, 0]
        x0 = p1[j, 1, 0]
        y0 = p1[j, 2, 0]
    else:
        x = p1[j, 1, 0]
        y = p1[j, 2, 0]
        x0 = p1[j, 3, 0]
        y0 = p1[j, 4, 0]
    length = p1[j, 8, 0]
    tx = [x, x0]
    ty = [y, y0]
    txPerp, tyPerp, tPLm, tPLc = getPL(tx, ty, (x + x0) / 2, (y + y0) / 2)

    ang = 1 if y - y0 > 0 else 2
    side = checkside(x, y, tPLm, tPLc, ang)

    pressure = p
    maxPressure = 0

    for i in range(len(t1)):
        isCon = isConnected(i, j, s, tPLm, tPLc, ang, side, m, c, side2)
        if (isCon==1 or isCon==2) and t1[i,1]:
            if maxPressure < t1[i, 1] and t1[i, 1] <= p:
                maxPressure = t1[i, 1]
                if isCon == 1:
                    l = getDist(i, j, s)
                else:
                    l = spatial.distance.pdist([[x0, y0], [t1[i, 3], t1[i, 4]]])

                pressure = ((length * maxPressure) + ((l - length) * p)) / l
                gradiantOut = (maxPressure-p) / l
                p1[j, 13, 0] = t1[i, 1]
                if p1[j, 0, 0] == 138:
                    agd = 1
                mode = 1
            if p1[j, 0, 0] == 138:
                agd = 1
            if maxPressure < t1[i, 1] and t1[i, 1] > p:
                pp = t1[i, 1]
                pipeId = p1[j, 0, 0]
                p1[j, 17, 0] = 5

    if maxPressure == 0:
        pressure = p + length * gradiant
        mode = 2
        gradiantOut = gradiant
        if p1[j, 17, 0] == 5:
            p1[j, 17, 0] = 2

    if pressure < 0:
        pressure = p
        mode = 2.1
        gradiantOut = gradiant
    return pressure, mode, gradiantOut


def getDist(div,pip,s):
    global p1
    global t1
    dist = 0
    if s == 1:
        for i in range(int(p1[pip, 18, 0])):
            if p1[pip, 19, i] == t1[div, 0]:
                dist = p1[pip, 22, i]
                break

    elif s == 2:
        for i in range(int(p1[pip, 20, 0])):
            if p1[pip, 21 , i] == t1[div, 0]:
                dist = p1[pip, 23, i]
                break
    return dist


def compute(x, y, p, m, c, side, length, gradiant):
    global p1
    global count2
    for i in range(len(p1)):
        if x == p1[i, 1, 0] and y == p1[i, 2, 0] and p1[i, 12, 0] == 0:
            p1[i, 12, 0] = 1
            p1[i, 7, 0], p1[i, 16, 0], g = findPressureBasic(i, 2, p, m, c, side, gradiant)
            fillSamePoints(p1[i, 3, 0], p1[i, 4, 0], p1[i, 7, 0], m, c, side)
            p1[i, 14, 0] = g
            compute(p1[i, 3, 0], p1[i, 4, 0], p1[i, 7, 0], m, c, side, p1[i, 8, 0], g)
        else:
            if x == p1[i, 3, 0] and y == p1[i, 4, 0] and p1[i, 11, 0] == 0:
                p1[i, 11, 0] = 1
                p1[i, 6, 0], p1[i, 15, 0], g  = findPressureBasic(i, 1, p, m, c, side, gradiant)
                fillSamePoints(p1[i, 1, 0], p1[i, 2, 0], p1[i, 6, 0], m, c, side)
                p1[i, 14, 0] = g
                compute(p1[i, 1, 0], p1[i, 2, 0], p1[i, 6, 0], m, c, side, p1[i, 8, 0], g)


def cleanStatus(side, m, c):
    global p1
    for i in range(len(p1)):
        if checkside(p1[i, 1, 0], p1[i, 2, 0], m, c, side) == 0 and p1[i, 0 , 0]!=80:
            p1[i, 11, 0]=0
            p1[i, 15, 0] = 0
        if checkside(p1[i, 3, 0], p1[i, 4, 0], m, c, side) == 0 and p1[i, 0 , 0]!=37:
            p1[i, 12, 0] = 0
            p1[i, 16, 0] = 0


def ontheLine(i, x, y):
    global p1
    status = 0
    if p1[i, 1, 0] > p1[i, 3, 0]:
        if p1[i, 1, 0] > x and p1[i, 3, 0] < x:
            status = 1
    else:
        if p1[i, 1, 0] < x and p1[i, 3, 0] > x:
            status = 1

    if p1[i, 2, 0] > p1[i, 4, 0]:
        if p1[i, 2, 0] < y and p1[i, 4, 0] > y:
            status = 0
    else:
        if p1[i, 2, 0] > y and p1[i, 4, 0] < y:
            status = 0

    return status


def addCrossPoints():
    ndivc = 0
    for m in range(len(p1)):
        cg = (p1[m, 6, 0] - p1[m, 7, 0]) / p1[m, 8, 0]
        if cg > 0:
            cg = -cg
        if cg < p1[m, 14, 0]:
            gdiff = cg - p1[m, 14, 0]
        else:
            gdiff = p1[m, 14, 0] - cg
        if (gdiff < -1 or gdiff > 1) and p1[m, 14, 0] != 0 and ~(
                p1[m, 1, 0] == pminX and p1[m, 2, 0] == pminY) and ~(p1[m, 3, 0] == pminX and p1[m, 4, 0] == pminY):
            pm = (p1[m, 2, 0] - p1[m, 4, 0]) / (p1[m, 1, 0] - p1[m, 3, 0])
            c = p1[m, 2, 0] - pm * p1[m, 1, 0]
            cx = (c - tPLc) / (tPLm - pm)
            cy = pm * cx + c
            s = ontheLine(m, cx, cy)
            if s == 1:
                len1 = spatial.distance.pdist([[p1[m, 1, 0], p1[m, 2, 0]],[cx, cy]])
                length = p1[m, 8, 0]
                len2 = length - len1
                cp = (len2 * p1[m, 6, 0] + len1 * p1[m, 7, 0]) / length
                tempx = p1[m, 3, 0]
                p1[m, 3, 0] = cx
                tempy = p1[m, 4, 0]
                p1[m, 4, 0] = cy
                p1[m, 8, 0] = len1
                n = len(p1) - 1
                p1[n, 0, 0] = -p1[m, 0, 0]
                p1[n, 1, 0] = cx
                p1[n, 2, 0] = cy
                p1[n, 3, 0] = tempx
                p1[n, 4, 0] = tempy
                p1[n, 8, 0] = len2
                nt = len(t1) - 1
                ndivc = ndivc + 1
                t1[nt, 0] = ndivc
                t1[nt, 1] = cp
                t1[nt, 3] = cx
                t1[nt, 4] = cy


def xDevice():
    for i in range(len(t1)):
        for j in range(len(p1)):
            if ((p1[j, 1, 0] - diff < t1[i, 3] < p1[j, 1, 0] + diff and p1[j, 2, 0] - diff < t1[i, 4] < p1[j, 2, 0] + diff) or (
                    p1[j, 3, 0] - diff < t1[i, 3] < p1[j, 3, 0] + diff and p1[j, 4, 0] - diff < t1[i, 4] < p1[j, 4, 0] + diff)) and p1[j, 10, 0] == 1:
                if p1[j, 1, 0] - diff < t1[i, 3] < p1[j, 1, 0] + diff and p1[j, 2, 0] - diff < t1[i, 4] < p1[j, 2, 0] + diff:
                    t1[i, 3] = p1[j, 1, 0]
                    t1[i, 4] = p1[j, 2, 0]

                if p1[j, 3, 0] - diff < t1[i, 3] < p1[j, 3, 0] + diff and p1[j, 4, 0] - diff < t1[i, 4] < p1[j, 4, 0] + diff:
                    t1[i, 3] = p1[j, 3, 0]
                    t1[i, 4] = p1[j, 4, 0]

                t1[i, 5] = 1
                break


def  findMinPoint():
    global t1
    global p1
    global mainLineT
    global mainLineT2
    global mainPipeMark
    global firstPipeId
    global firstMX
    global firstMY
    found = 0
    x = 0
    y = 0
    n = 1
    for i in range(len(t1)):
        p1[:, 9, 0] = mainPipeMark
        x1, y1 = findMainLinePoint(t1[i, 3], t1[i, 4])
        t1[i, 7] = x1
        t1[i, 8] = y1

        pressure = t1[i, 1]
        for j  in range(len(mainLineT)):
            if mainLineT[j, 0] == x1  and mainLineT[j, 1] == y1:
                found = 1
                if mainLineT[j, 2] < pressure:
                    mainLineT[j, 2] = pressure
                    mainLineT[j, 3] = t1[i, 0]

        if found == 0:
            n = n + 1
            mainLineT[n, 0] = x1
            mainLineT[n, 1] = y1
            mainLineT[n, 2] = pressure
            mainLineT[n, 3] = t1[i, 0]
        found = 0

    p1[:, 9, 0] = mainPipeMark
    for k in range(len(mainLineT)):
        p1[:, 17, 0] = 0
        findDistance(firstMX, firstMY, mainLineT[k, 0], mainLineT[k, 1], 0, k)
    p1[:, 17, 0] = 0
    m = (mainLineT[0, 2] - mainLineT[1, 2]) / (mainLineT[0, 4] - mainLineT[1, 4])
    c = mainLineT[0, 2] - (m * mainLineT[0, 4])
    counter = 0

    temp = np.zeros([58,6])

    for i in range(len(mainLineT)):
        if mainLineT[i, 2] !=0 and checkside(mainLineT[i, 4], mainLineT[i, 2], m, c, 1) == 0:
            temp[counter,:] = mainLineT[i,:]
            counter = counter + 1

    counter = 1
    k = 0
    temp[:, 5] = 0
    for i in range(2,len(temp)):
        if temp[i,4]!=0:
            if spatial.distance.pdist([[temp[0, 4], 0 ], [temp[i, 4] ,0]]) <= spatial.distance.pdist([[0 ,temp[1, 4]],[0 ,temp[i, 4]]]):
                tk = spatial.distance.pdist([[temp[0, 2], 0 ], [temp[i, 2], 0]]) / spatial.distance.pdist([[temp[0, 4] ,0 ], [temp[i, 4],0]])
                k = k + tk
                temp[i, 5] = tk
                counter = counter + 1
            else:
                tk = spatial.distance.pdist([[temp[1, 2], 0], [temp[i, 2], 0]]) / spatial.distance.pdist( [[temp[1, 4], 0], [temp[i, 4], 0]])
                k = k + tk
                temp[i, 5] = tk
                counter = counter + 1
    k = k / counter
    counter = 0

    for i in range(len(temp)):
        if k > temp[i, 5]:
            mainLineT2[counter, :]=temp[i,:]
            counter = counter + 1

    for i in range(58):
        if mainLineT2[i][4] == 0:
            mainLineT3 = mainLineT2[:i]
            break

    # print(mainLineT3)

    pol = np.polyfit(mainLineT3[:, 4], mainLineT3[:, 2], 4)
    fitcX = np.linspace(min(mainLineT3[:, 4]), max(mainLineT3[:, 4]), 50)
    fitcY = np.polyval(pol, fitcX)

    ymin = max(fitcY)
    yminI = 0
    fitcX = fitcX.T

    for i in range(len(fitcX)):
        if fitcX[i] > temp[0, 4]  and  fitcX[i] < temp[1, 4]  and  fitcY[i] <= ymin :
            ymin = fitcY[i]
            yminI = i

    diff = max(fitcX)

    for i in range(len(fitcX)):
        tempDiff = spatial.distance.pdist([[fitcX[yminI], 0], [mainLineT[i, 4], 0]])
        if tempDiff < diff:
            diff = tempDiff
            x = mainLineT[i, 0]
            y = mainLineT[i, 1]
    return x , y


def findMainLinePoint(x1, y1):
    global p1
    x = 0
    y = 0
    for i in range(len(p1)):
        if (p1[i, 1, 0] == x1 and  p1[i, 2, 0] == y1)  and  p1[i, 9, 0] !=2:
            if (p1[i, 9, 0] == 1):
                x = x1
                y = y1
                break
            else:
                p1[i, 9, 0] = 2
                for j in range(len(p1)):
                    if (p1[i, 3, 0] == p1[j, 1, 0] and  p1[i, 4, 0] == p1[j, 2, 0]) and  p1[j, 9, 0]!=2:
                        if p1[j, 9, 0] == 1:
                            x = p1[i, 3, 0]
                            y = p1[i, 4, 0]
                            break

                    elif (p1[i, 3, 0] == p1[j, 3, 0] and  p1[i, 4, 0] == p1[j, 4, 0]) and  p1[j, 9, 0]!=2:
                        if p1[j, 9, 0] == 1:
                            x = p1[i, 3, 0]
                            y = p1[i, 4, 0]
                            break

                if x!= 0 and y!= 0:
                    break

                for j in range(len(p1)):
                    if (p1[i, 3, 0] == p1[j, 1, 0] and  p1[i, 4, 0] == p1[j, 2, 0]) and  p1[j, 9, 0]!=2:
                        x, y = findMainLinePoint(p1[j, 3, 0], p1[j, 4, 0])
                    elif(p1[i, 3, 0] == p1[j, 3, 0] and  p1[i, 4, 0] == p1[j, 4, 0]) and  p1[j, 9, 0]!=2:
                        x, y = findMainLinePoint(p1[j, 1, 0], p1[j, 2, 0])
                    if x != 0 and y != 0:
                        break

        if (p1[i, 3, 0] == x1 and p1[i, 4, 0] == y1) and p1[i, 9, 0] != 2:
            if (p1[i, 9, 0] == 1):
                x = x1
                y = y1
                break
            else:
                p1[i, 9, 0 ] = 2
                for j in range(len(p1)):
                    if (p1[i, 1, 0] == p1[j, 1, 0] and p1[i, 2, 0] == p1[j, 2, 0]) and p1[j, 9, 0] != 2:
                        if p1[j, 9, 0] == 1:
                            x = p1[i, 1, 0]
                            y = p1[i, 2, 0]
                            break

                    elif (p1[i, 1, 0] == p1[j, 3, 0] and p1[i, 2, 0] == p1[j, 4, 0]) and p1[j, 9, 0] != 2:
                        if p1[j, 9, 0] == 1:
                            x = p1[i, 1, 0]
                            y = p1[i, 2, 0]
                            break
                if x != 0 and y != 0:
                    break

                for j in range(len(p1)):
                    if (p1[i, 1, 0] == p1[j, 1, 0] and p1[i, 2, 0] == p1[j, 2, 0]) and p1[j, 9, 0] != 2:
                        x, y = findMainLinePoint(p1[j, 3, 0], p1[j, 4, 0])
                    elif (p1[i, 1, 0] == p1[j, 3, 0] and p1[i, 2, 0] == p1[j, 4, 0 ]) and p1[j, 9, 0] != 2:
                        x, y = findMainLinePoint(p1[j, 1, 0], p1[j, 2, 0])
                    if x != 0 and y != 0:
                        break
    return x, y

def findDistance(x0,y0,x,y,length,k):
    global p1
    global mainLineT

    for i  in range(len(p1)):
        if p1[i, 17, 0] != 6:
            if x0==p1[i, 1, 0] and  y0==p1[i, 2, 0]  and  x==p1[i, 3, 0] and y==p1[i, 4, 0]:
                mainLineT[k,4]=length+p1[i, 8, 0]
                break
            elif  x0==p1[i, 3, 0] and  y0==p1[i, 4, 0]  and  x==p1[i, 1, 0] and y==p1[i, 2, 0]:
                mainLineT[k, 4] = length + p1[i, 8, 0]
                break

            if x0 == p1[i, 1, 0 ] and y0 == p1[i, 2, 0]:
                p1[i, 17, 0] = 6
                length2 = length +p1[i,8,0]
                findDistance(p1[i, 3, 0],p1[i, 4, 0],x,y,length2,k)

            elif x0 == p1[i, 3, 0 ] and y0 == p1[i, 4, 0]:
                p1[i, 17, 0] = 6
                length2 = length+p1[i,8,0]
                findDistance(p1[i, 1, 0],p1[i, 2, 0],x,y,length2,k)



def resetTable(pdata):
    global p1
    global t2
    p1[:, 6:8, :] = 0
    p1[:, 10:13, 0] = 0
    p1[:, 17:24, 0] = 0
    gid1 = t2['BUZ_NUMBER'].values
    gid2 = pdata.index

    for i in range(len(gid1)):
        for j in range(len(gid2)):
            if str(gid1[i]).zfill(8) == gid2[j]:
                t2.iloc[i, 1] = pdata.iloc[j]
            elif 'FA' + str(gid1[i]).zfill(6) == gid2[j]:
                t2.iloc[i, 1] = pdata.iloc[j]

            if gid2[j] == '00G08003':
                pB = pdata.iloc[j]
            elif gid2[j] == '00G08037':
                pB2 = pdata.iloc[j]
        t2 = t2.fillna(0)
    return p1, t2, pB, pB2



def insert_valve(time,data):
    db = oracle.connect('pipegis/pipegis@192.168.1.55/ORCL')
    db.autocommit = True
    cur = db.cursor()
    d = pd.read_excel('pipeline717.xlsx')
    gid = data.iloc[:,0]
    p_in = data.iloc[:,6]
    p_out = data.iloc[:,7]
    lenth = data.iloc[:,8]
    Diameter = d['diameter'].values
    delt_p = abs((p_in - p_out) / lenth)

    for i in range(len(data)):

        q = 0.0385 * ((abs(p_out[i] ** 2 - p_in[i] ** 2) * (Diameter[i] ** 5) / lenth[i]) ** 0.5) * (10 ** -4.5)
        if 0 < delt_p[i] <= 1:
            l = 1
        elif 1 < delt_p[i] < 2.5:
            l = 2
        else:
            l = 3
        if p_in[i] < 240 and p_out[i] < 240:
            l = 4

        param = {'id': int(gid[i]), 'tim': time, 'sl': l, 't0': p_in[i]/1000 , 't1': p_out[i]/1000, 'q': q, 'sign': delt_p[i]}
        cur.execute('insert into PIPE_DEMO(GID,DATETIM,SIGN_LEVEL,T0,T1,Q,SIGN) values(:id,:tim,:sl,:t0,:t1,:q ,:sign)', param)

    cur.close()
    db.close()


if __name__ == '__main__':

    t2 = pd.read_excel('device0710.xls', sheet_name='device')
    '''pandas 取X,Y'''
    X = t2['X'].values
    Y = t2['Y'].values

    p_data = pd.read_csv('bq_data.csv')
    px, py = -25756.76, 20285.65  # 宝钱 00G08003
    px2, py2 = -20152.4087, 11812.0661  # 永盛 00G08037
    l1 = 58.071283
    l2 = 3.919698
    diff = 0.5
    LineWidth = 2

    tx = [px, px2]
    ty = [py, py2]
    tx1, ty1, tm1, tc1 = getPL(tx, ty, px, py)
    tx2, ty2, tm2, tc2 = getPL(tx, ty, px2, py2)

    p1 = pd.read_excel('pipeline717.xlsx', sheet_name='Export_Output')
    p2 = np.array(p1)
    p1 = np.zeros((len(p2), 24, 100), dtype=float)

    mainLineT = np.zeros([58, 6], dtype=float)
    mainLineT2 = np.zeros([58, 6], dtype=float)

    '''二维数据赋予三维'''
    for i in range(len(p2)):
        for j in range(24):
            p1[i, j, 0] = p2[i, j]

    for a in range(96):
        mainPipeMark = copy.deepcopy(p1[:, 9, 0])
        pmin = 256509
        p_list = p_data.iloc[a, :]
        p_time = p_data.iloc[a, 0]
        p_time = datetime.datetime.strptime(p_time, '%Y-%m-%d %H:%M:%S')

        p1, t2, pB, pB2 = resetTable(p_list)
        t1 = np.array(t2)

        plotPipesA(px, py, diff)
        xDevice()

        firstPipeId = 174
        firstMX = -27733.114
        firstMY = 23458.62
        mainLineT[0, 0], mainLineT[0, 1]= findMainLinePoint(px, py)
        mainLineT[0, 2] = pB
        mainLineT[0, 3] = 1
        p1[:, 9, 0]= mainPipeMark
        mainLineT[1, 0] , mainLineT[1, 1]= findMainLinePoint(px2, py2)
        mainLineT[1, 2] = pB2
        mainLineT[1, 3] = 2
        pminX, pminY = findMinPoint()

        xFit, yFit = getFitMainLine(X, Y)
        txPerp, tyPerp, tPLm, tPLc = getPL(xFit, yFit, pminX, pminY)

        mainPx = np.linspace(-40000, -5000, 10)
        mainPy = tPLm * mainPx + tPLc
        p1[:, 10:17, 0] = 0
        p1[:, 6, 0] = 0
        p1[:, 7, 0] = 0
        plotPipesA(px, py, diff)
        find_connection()

        for j in range(len(p1)):
            if p1[j, 0, 0] == 80 :
                p1[j, 6, 0] = pB
                p1[j, 11, 0] = 1
            if  p1[j, 0, 0] == 37:
                p1[j, 7, 0] = pB2
                p1[j, 12, 0] = 1
        ndivc = 0
        compute(px, py, pB, tPLm, tPLc, 1, l1, 0)
        cleanStatus(1, tPLm, tPLc)
        compute(px2, py2, pB2, tPLm, tPLc, 2, l2, 0)

        # addCrossPoints()
        # p1[:, 10:24, :]=0
        # p1[:, 6, 0] = 0
        # p1[:, 7, 0] = 0
        #
        # plotPipesA(px, py, diff)
        # find_connection()
        #
        # for j in range(len(p1)):
        #     if p1[j, 0, 0] == 80 or p1[j, 0, 0] == 37:
        #         p1[j, 6, 0] = pB
        #         p1[j, 11, 0] = 1
        #
        # compute(px, py, pB, tPLm, tPLc, 1, l1, 0)
        #
        # cleanStatus(1, tPLm, tPLc)
        # compute(px2, py2, pB2, tPLm, tPLc, 2, l2, 0)

        p_print = pd.DataFrame(p1[:, 0:17, 0])
        insert_valve(p_time, p_print)