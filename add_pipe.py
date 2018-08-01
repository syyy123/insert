
#coding:'utf-8'

import pandas as pd
import numpy as np
from scipy import spatial
import copy
import datetime
import matplotlib.pyplot as plt
import cx_Oracle as oracle


'''递归查找坐标，解决坐标误差问题'''
'''to adjust the original coordinates of pipes in a one pipe begins at an end of another pipe(to connect the pipe line 
system) xB,yB are two points to start connecting form and diff is a +/-difference for errors '''

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

'''to set the status(which side the device belongs to) before making connection table
x,y are the coordinates of the pipe end in the direction that it should traverse while x0,y0 are the other ends.
 side is the end of the pipe(1 or 2), cp is the index of the current pipe, side2 is the side of the system( which main 
 point are we working on)(1-top or 2-bottom)'''
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

'''to make the connection table / to call the getConnection function for every pipe with the proper initial values'''
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

'''to make the connection table
x,y are the coordinates of the pipe end in the direction that it should traverse while x0,y0 are the other ends.
 side is the end of the pipe(1 or 2),i is the index of the pipe that we are finding connections while cp is the index
  of the current pipe, length is the distance from the end of the pipe to the device(always begin with 0 and increase
   along the traverse) side2 is the side of the system( which main point are we working on)(1-top or 2-bottom)
'''
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

'''to get a  perpendicular line to a line
x,y are coordinates of the point along the line,x0,y0 are the coordinates of the point where the PL need to cross'''
def getPL(x, y, x0, y0):
    m =(y[-1] - y[0]) / (x[-1] - x[0])
    PLm = np.tan(np.arctan(m) + np.pi / 2)
    PLc = y0 - PLm * x0
    plX = x
    plY = [PLm * i + PLc for i in plX]

    return plX, plY, PLm, PLc

'''to get a fit line through all the xIn,yIn coordinates, those are arrays'''
def getFitMainLine(xIn, yIn):
    linearCoefficients = np.polyfit(xIn, yIn, 1)
    x = np.linspace(min(xIn), max(xIn), 50)
    y = np.polyval(linearCoefficients, x)

    return x, y

'''when the pressure of an end of a pipes is found, this function is to assign the pressure of that point to all the end 
of the pipes which are connected to it
x,y are the coordinates of the point and p is pressure. m,c,side are related to the checkside function'''
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

'''to check whether a point in the the given side form the given line
(x,y) is the point, m is gradient and c is constant of the line eqn, side the above y axis(1) or not'''
def checkside(x, y, m, c, angle):
    if angle == 1:
        isFront = 1 if y > m * x + c else 0
    else:
        isFront = 1 if y < m * x + c else 0

    return isFront

'''to check whether a device is connected to the given pipe or not 
div and pip are the indexes of device and pipe. s is to use the connection table s is the end of the pipe(1 or 2)
the rest of the arguments are to use the method of perpendicular line to assume the connectivity (for the pipes where 
one end is connected to the other)m,c,m2,c2 are the m and c of line across the pipe and line across the system
respectively. ang is to indicate the angle of the pipe. side and side2 are to indicate the sides  where device is
 supposed to be from the lines across the pipe and line across the system respectively. 
'''
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

'''to calculate the pressure of a point
j is the index of the pipe where s indicates the end(1 or 2) of pipe. p is the pressure of the previous point(we are 
traversing the pipe system starting from a main point and outwards).m and c are gradient and constant. side to is the
current potion or the side of the pipe system that we are working on. gradient is the pressure gradient or the pressure 
drop of the previous pipe  '''
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

'''to get the distance from the pipe to a device
div and pip are the pipe index and that of device relatively, s is the end(1 or 2) of the pipe to look connection from.
the distance is saved to table at he same time along with the connections(getConnection function)'''
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

'''to compute pressure of the system, to travers the pipe system and call the findPressureBasic function on all the 
pipes along the way
x,y are the coordinates the point where travers is headed. p is the pressure of the previous point.
other arguments are for the findPressureBasic function and they are explained there '''
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


'''to remove the compute status of the pipes from the other side of the system
side is side of the system, m and c are the gradient and constant of the mid line'''
def cleanStatus(side, m, c):
    global p1
    for i in range(len(p1)):
        if checkside(p1[i, 1, 0], p1[i, 2, 0], m, c, side) == 0 and p1[i, 0 , 0]!=80:
            p1[i, 11, 0]=0
            p1[i, 15, 0] = 0
        if checkside(p1[i, 3, 0], p1[i, 4, 0], m, c, side) == 0 and p1[i, 0 , 0]!=37:
            p1[i, 12, 0] = 0
            p1[i, 16, 0] = 0

'''to check if a give point is on a give a pipe
i is pipe index and x and y are coordinates of the point '''
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

'''to brake the pipes in two and add the average pressure to the brake point as a device if the pipe is crossed
by the mid line(to make that value common for both sides of the system)'''
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

'''it is to adjust the coordinates of the devices so that the are connected to an end of a pipe'''
'''将t1表中每个设备坐标和管道坐标进行比较，并将寻找到的管道坐标替换设备坐标，mark找到的设备为1'''
def xDevice():
    for i in range(len(t1)):
        for j in range(len(p1)):
            '''plotPipeA标记p1[j, 10, 0] == 1'''
            if p1[j, 10, 0] == 1:
                if p1[j, 1, 0] - diff < t1[i, 3] < p1[j, 1, 0] + diff and p1[j, 2, 0] - diff < t1[i, 4] < p1[j, 2, 0] + diff:
                    t1[i, 3] = p1[j, 1, 0]
                    t1[i, 4] = p1[j, 2, 0]
                    t1[i, 5] = 1
                    break
                if p1[j, 3, 0] - diff < t1[i, 3] < p1[j, 3, 0] + diff and p1[j, 4, 0] - diff < t1[i, 4] < p1[j, 4, 0] + diff:
                    t1[i, 3] = p1[j, 3, 0]
                    t1[i, 4] = p1[j, 4, 0]
                    t1[i, 5] = 1
                    break

'''to find the minimum pressure point between the two main out points'''
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
    '''remove the pressure 0 devices'''
    for i in range(len(mainLineT)):
        if mainLineT[i, 2] !=0 and checkside(mainLineT[i, 4], mainLineT[i, 2], m, c, 1) == 0:
            temp[counter,:] = mainLineT[i,:]
            counter = counter + 1

    counter = 1
    k = 0
    temp[:, 5] = 0
    '''to use an average k (a connection between distance and  difference of pressure with the main points)
        to remove unusual or potentially faulty readings'''
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
            mainLineT2[counter, :] = temp[i,:]
            counter = counter + 1

    for i in range(58):
        if mainLineT2[i][4] == 0:
            mainLineT3 = mainLineT2[:i]
            break

    # print(mainLineT3)
    '''get the polyfit curve to find a min point'''
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

'''to find the coordinates of place where given device is connected to the mainPipe(mainLine)
x1,y1 are the coordinates of the device'''
def findMainLinePoint(x1, y1):
    x = 0
    y = 0
    for i in range(len(p1)):
        if p1[i, 1, 0] == x1 and  p1[i, 2, 0] == y1  and  p1[i, 9, 0] != 2:
            if p1[i, 9, 0] == 1:
                x = x1
                y = y1
                break
            else:
                p1[i, 9, 0] = 2
                for j in range(len(p1)):
                    if p1[i, 3, 0] == p1[j, 1, 0] and  p1[i, 4, 0] == p1[j, 2, 0] and  p1[j, 9, 0] != 2:
                        if p1[j, 9, 0] == 1:
                            x = p1[i, 3, 0]
                            y = p1[i, 4, 0]
                            break

                    elif p1[i, 3, 0] == p1[j, 3, 0] and  p1[i, 4, 0] == p1[j, 4, 0] and  p1[j, 9, 0] != 2:
                        if p1[j, 9, 0] == 1:
                            x = p1[i, 3, 0]
                            y = p1[i, 4, 0]
                            break

                if x!= 0 and y!= 0:
                    break

                for j in range(len(p1)):
                    if p1[i, 3, 0] == p1[j, 1, 0] and  p1[i, 4, 0] == p1[j, 2, 0] and  p1[j, 9, 0] != 2:
                        x, y = findMainLinePoint(p1[j, 3, 0], p1[j, 4, 0])
                    elif p1[i, 3, 0] == p1[j, 3, 0] and  p1[i, 4, 0] == p1[j, 4, 0] and  p1[j, 9, 0] != 2:
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



'''to find the distance between two point in the pipe system
x0,y0 are the starting point and x,y are the coordinates of the destination length is current length or distance
traversed (it always starts with 0 and increase with traverse, k is the index of the table need saving,
mainLineT index'''
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
                length2 = length +p1[i, 8, 0]
                findDistance(p1[i, 3, 0],p1[i, 4, 0],x,y,length2,k)

            elif x0 == p1[i, 3, 0 ] and y0 == p1[i, 4, 0]:
                p1[i, 17, 0] = 6
                length2 = length+p1[i,8,0]
                findDistance(p1[i, 1, 0],p1[i, 2, 0],x,y,length2,k)


def findPoint(x, y, x0, y0,i):
    global p1
    found = 0
    p, g, side, id = 0, 0, 0, 0


    if x == p1[i, 1, 0] and y == p1[i, 2, 0] :
        p = p1[i, 6, 0]
        g = p1[i, 14, 0]
        found = 1
        side = 1
        id = p1[i, 0, 0]

    elif x == p1[i, 3, 0] and y == p1[i, 4, 0] :
        p = p1[i, 7, 0]
        g = p1[i, 14, 0]
        found = 1
        side = 2
        id = p1[i, 0, 0]

    if found == 0:
        for j in range(len(p1)):
            if x0 == p1[j, 1, 0] and y0 == p1[j, 2, 0]  and  p1[j, 17, 0] == 0 :
                p1[j, 17, 0] = 1
                p, g, found, side, id = findPoint(x, y, p1[j, 3, 0], p1[j, 4, 0],j)

            elif x0 == p1[j, 3, 0] and y0 == p1[j, 4, 0] and  p1[j, 17, 0] == 0:
                p1[j, 17, 0] = 1
                p, g, found, side, id = findPoint(x, y, p1[j, 1, 0], p1[j, 2, 0],j)
            if found == 1:
                break
    return p, g, found, side, id


def removeDivAfter(id, side):
    global p1
    global t1
    for i in range(len(p1)):
        if p1[i, 0, 0] == id:
            if side == 1:
                no = p1[i, 18, 0]
                for j in range(int(no)):
                    div = p1[i, 19, j]
                    for k in range(len(t1)):
                        if t1[k, 0] == div:
                            t1[k, 1] = np.nan
                            break

            else:
                no = p1[i, 20, 0]
                for j in range(int(no)):
                    div = p1[i, 21, j]
                    for k in range(len(t1)):
                        if t1[k, 0] == div:
                            t1[k, 1] = np.nan
                            break
            break

def add_values(name , i, j, value):
    if name.shape[0]-1 < i :
        namet = np.zeros([i+1 , name.shape[1]])
        print(namet.shape)
        namet[0:name.shape[0] , 0:name.shape[1]] = name
        namet[i , j] = value
        return namet
    else:
        name[i, j] = value
        return name


def addImaginaryPipe(x1,y1,x2,y2):
    global p1
    global t1
    global d1

    p1[:, 17, 0] = 0
    tp1,tg1,f,side1,pid1 = findPoint(x1,y1,px,py,0)
    print( tp1,tg1,f,side1,pid1)
    p1[:, 17, 0] = 0
    tp2,tg2,f,side2,pid2 = findPoint(x2,y2,px,py,0)
    print(tp2,tg2,f,side2,pid2)

    g = (tg1+tg2)/2
    length = spatial.distance.pdist([[x1,y1],[x2,y2]])
    pressure1 = (tp1+tp2-(g*length))/2
    pressure2 = (tp1+tp2+(g*length))/2
    it = len(t1)
    # print(it)

    removeDivAfter(pid1,side1)
    removeDivAfter(pid2,side2)

    # print(t1.shape[1])

    t1 = add_values(t1, it, 0, 1)
    t1 = add_values(t1, it, 1, pressure1)
    t1 =add_values(t1, it, 3, x1)
    t1 =add_values(t1, it, 4, y1)


    it = it + 1
    t1 = add_values(t1, it, 0, 2)
    t1 = add_values(t1, it, 1, pressure2)
    t1 = add_values(t1, it, 3, x2)
    t1 = add_values(t1, it, 4, y2)

'''to clear the computed data and status variables, to initialize tables for the next data set'''
'''初始化数据，返回p1管道信息，t2调压器压力信息，两门站压力pB,pB2'''
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


#
# def insert_valve(time,data):
#     db = oracle.connect('pipegis/pipegis@192.168.1.55/ORCL')
#     db.autocommit = True
#     cur = db.cursor()
#     d = pd.read_excel('/home/zhangxuan/insert2/pipeline717.xlsx')
#     gid = data.iloc[:,0]
#     p_in = data.iloc[:,6]
#     p_out = data.iloc[:,7]
#     lenth = data.iloc[:,8]
#     Diameter = d['diameter'].values
#     delt_p = abs((p_in - p_out) / lenth)
#
#     for i in range(len(data)):
#
#         q = 0.0385 * ((abs(p_out[i] ** 2 - p_in[i] ** 2) * (Diameter[i] ** 5) / lenth[i]) ** 0.5) * (10 ** -4.5)
#         if 0 < delt_p[i] <= 1:
#             l = 1
#         elif 1 < delt_p[i] < 2.5:
#             l = 2
#         else:
#             l = 3
#         if p_in[i] < 240 and p_out[i] < 240:
#             l = 4
#
#         param = {'id': int(gid[i]), 'tim': time, 'sl': l, 't0': p_in[i]/1000 , 't1': p_out[i]/1000, 'q': q, 'sign': delt_p[i]}
#         cur.execute('insert into PIPE_DEMO(GID,DATETIM,SIGN_LEVEL,T0,T1,Q,SIGN) values(:id,:tim,:sl,:t0,:t1,:q ,:sign)', param)
#
#     cur.close()
#     db.close()


if __name__ == '__main__':
    t2 = pd.read_excel('device0710.xls', sheet_name='device')
    '''pandas 取X,Y'''
    X = t2['X'].values
    Y = t2['Y'].values

    '''p_data一整天调压器数据'''
    p_data = pd.read_csv('bq_data.csv')
    px, py = -25756.76, 20285.65  # 宝钱 00G08003
    px2, py2 = -20152.4087, 11812.0661  # 永盛 00G08037
    l1 = 58.071283 #宝钱管长
    l2 = 3.919698 #永盛管长
    diff = 0.5
    LineWidth = 2

    tx = [px, px2]
    ty = [py, py2]
    tx1, ty1, tm1, tc1 = getPL(tx, ty, px, py) #过（px，py)，垂直于两门站连线的直线
    tx2, ty2, tm2, tc2 = getPL(tx, ty, px2, py2) #过（px2，py2)，垂直于两门站连线的直线

    p1 = pd.read_excel('pipeline717.xlsx', sheet_name='Export_Output')
    p2 = np.array(p1)
    p1 = np.zeros((len(p2), 24, 100), dtype=float)

    mainLineT = np.zeros([58, 6], dtype=float)
    mainLineT2 = np.zeros([58, 6], dtype=float)

    '''二维数据赋予三维,扩展维度存储数据'''
    for i in range(len(p2)):
        for j in range(24):
            p1[i, j, 0] = p2[i, j]


    for a in range(1):
        mainPipeMark = copy.deepcopy(p1[:, 9, 0])
        pmin = 256509
        p_list = p_data.iloc[a, :]
        p_time = p_data.iloc[a, 0]
        p_time = datetime.datetime.strptime(p_time, '%Y-%m-%d %H:%M:%S')

        p1, t2, pB, pB2 = resetTable(p_list)
        t1 = np.array(t2)

        plotPipesA(px, py, diff)
        xDevice() #device 坐标和管道坐标绑定

        firstPipeId = 174
        firstMX = -27733.114
        firstMY = 23458.62
        mainLineT[0, 0], mainLineT[0, 1]= findMainLinePoint(px, py)
        mainLineT[0, 2] = pB
        mainLineT[0, 3] = 1
        p1[:, 9, 0] = mainPipeMark
        mainLineT[1, 0] , mainLineT[1, 1]= findMainLinePoint(px2, py2)
        mainLineT[1, 2] = pB2
        mainLineT[1, 3] = 2


        pminX, pminY = findMinPoint()
        xFit, yFit = getFitMainLine(X, Y)
        txPerp, tyPerp, tPLm, tPLc = getPL(xFit, yFit, pminX, pminY)
        mainPx = np.linspace(-40000, -5000, 10)
        mainPy = tPLm * mainPx + tPLc
        t1[:, 5] = 0
        p1[:, 10:24, 0] = 0
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

        p_print = pd.DataFrame(p1[:, 0:17, 0])
        p_print.to_excel(str(a) + '_3.xlsx')
        # insert_valve(p_time, p_print)

        addImaginaryPipe(-24885.0112,19983.5021 , -23833.8254, 19404.4631)
        t1[:, 5] = 0
        p1[:, 10:25, :] = 0
        p1[:, 6, 0] = 0
        p1[:, 7, 0] = 0
        plotPipesA(px, py, diff)
        find_connection()

        for j in range(len(p1)):
            if p1[j, 0, 0] == 80:
                p1[j, 6, 0] = pB
                p1[j, 11, 0] = 1
            if p1[j, 0, 0] == 37:
                p1[j, 7, 0] = pB2
                p1[j, 12, 0] = 1
        ndivc = 0
        compute(px, py, pB, tPLm, tPLc, 1, l1, 0)
        cleanStatus(1, tPLm, tPLc)
        compute(px2, py2, pB2, tPLm, tPLc, 2, l2, 0)

        p_print = pd.DataFrame(p1[:, 0:17, 0])
        p_print.to_excel(str(a) + '_4.xlsx')