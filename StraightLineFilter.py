import math as m
import numpy as np
import numpy.random as random

import threading
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

N = 50
Nlines = 0
pnts = np.zeros([3, N])
pntsBuf = np.zeros([2, N])
lines = np.zeros([4, N - 1])

tol = 0.1
mess = 0.1
shape = 0

fig = plt.figure()
ax = plt.axes([0.07, 0.25, 0.45, 0.7])

mutex = threading.RLock()

def lineApproxAveraging(pnts : np.ndarray, fr : int, to : int):

    b0 = np.exp(np.linspace(-0.25, 1 - 0.25, (to - fr))**2 / -0.03)
    # b0 = np.linspace(1.0, 0.0, N)**2.0
    b0 /= np.sum(b0)

    b1 = np.flip(b0)

    pnt0 = [np.sum(b0 * pnts[0, fr : to]), np.sum(b0 * pnts[1, fr : to])]
    pnt1 = [np.sum(b1 * pnts[0, fr : to]), np.sum(b1 * pnts[1, fr : to])]

    a = (pnt1[1] - pnt0[1]) / (pnt1[0] - pnt0[0])
    b = pnt0[1] - a * pnt0[0]
    return (a, b, np.mean((a * pnts[0, fr : to] - pnts[1, fr : to] + b)**2) / (b**2 + 1))

def lineApproxLMS(pnts : np.ndarray, fr : int, to : int):
    #LMS
    a = 0
    b = 0
    return (a, b, 0)

def firstPnt(pnts : np.ndarray) -> None:
    pnts[0, 0] = 0.5 * random.rand() - 0.25
    pnts[1, 0] = 0.5 * random.rand() - 0.25
    pnts[2, 0] = 2.0 * m.pi * random.rand() - m.pi

def createPnts(pnts : np.ndarray, N, d0 = 0.1, shape = 0, mess = 0.1) -> None:
    i_ang = 0
    deltaAng = 0.2 * random.rand() - 0.

    for i in range(1, N):
        d = d0 * (1 + random.randn() * mess)
        pnts[0, i] = pnts[0, i - 1] + d * m.cos(pnts[2, i - 1])
        pnts[1, i] = pnts[1, i - 1] + d * m.sin(pnts[2, i - 1])

        if (shape == 0):    #polyline
            if (random.rand() > 1 - 5.0 / N): # 5 fractures in average
                pnts[2, i] = pnts[2, i - 1] + m.pi * random.rand() - m.pi/2
                i_ang = i
            else:
                pnts[2, i] = pnts[2, i_ang] * (1 + random.randn() * mess)
        elif (shape == 1):  #circle
            pnts[2, i] = pnts[2, i - 1] + deltaAng

def getLines(lines : np.ndarray, pnts : np.ndarray, Npnts, tolerance = 0.1) -> int:
    """#returns the number of the gotten lines in lines"""

    global Nlines

    line = np.zeros([4])
    pcross = np.array([0.0, 0.0])

    i = 1
    Nlines = 0

    while i < Npnts:
        gap = tolerance
        i0 = i

        while True:

            line[0] = (pnts[1, i] - pnts[1, i - 1]) / (pnts[0, i] - pnts[0, i - 1])
            line[1] = pnts[1, i] - line[0] * pnts[0, i]
            byNpnts = 2

            while True:
                
                i += 1

                if (i < N and abs(line[0] * pnts[0, i] - pnts[1, i] + line[1]) / m.sqrt(line[0]**2 + 1) < gap):
                    if (not byNpnts % 2):
                        line[0] = (pnts[1, i - byNpnts // 2] - pnts[1, i - byNpnts]) / (pnts[0, i - byNpnts // 2] - pnts[0, i - byNpnts])
                        line[1] = pnts[1, i - byNpnts] - line[0] * pnts[0, i - byNpnts]
                    byNpnts += 1
                else:
                    (line[0], line[1], q0) = lineApproxAveraging(pnts, i - byNpnts, i)

                    while (q0 > 0.0001):
                        (line_0, line_1, q) = lineApproxAveraging(pnts, i - byNpnts, i - 1)
                        if (q > q0):
                            break
                        else:
                            i -= 1
                            byNpnts -= 1
                            line[0] = line_0
                            line[1] = line_1
                            q0 = q

                    if (Nlines > 0):
                        pcross[0] = (line[1] - lines[1, Nlines - 1]) / (lines[0, Nlines - 1] - line[0])
                        pcross[1] = line[0] * pcross[0] + line[1]

                        if (np.linalg.norm(pnts[:2, i - byNpnts] - pcross) > tolerance or m.isnan(pcross[0]) or m.isinf(pcross[0])):
                            if (byNpnts <= 2):
                                pcross[0] = (pnts[0, i - 2] + lines[0, Nlines - 1] * pnts[1, i - 2] - lines[0, Nlines - 1] * lines[1, Nlines - 1]) / (lines[0, Nlines - 1]**2 + 1)
                                pcross[1] = lines[0, Nlines - 1] * pcross[0] + lines[1, Nlines - 1]

                                line[0] = (pnts[1, i - 1] - pcross[1]) / (pnts[0, i - 1] - pcross[0])
                                line[1] = pcross[1] - line[0] * pcross[0]
                                lines[3, Nlines - 1] = pcross[0]
                                line[2] = pcross[0]
                            else:
                                i = i0
                                gap *= 0.75
                                break
                        else:
                            lines[3, Nlines - 1] = pcross[0]
                            line[2] = pcross[0]

                    else:
                        line[2] = (pnts[0, 0] + line[0] * pnts[1, 0] - line[0] * line[1]) / (line[0]**2 + 1)

                    if (i > N - 1):
                        line[3] = (pnts[0, N - 1] + line[0] * pnts[1, N - 1] - line[0] * line[1]) / (line[0]**2 + 1)

                    break

            if (i > i0):
                break
            else:
                continue

        lines[:, Nlines] = line
        Nlines += 1
    
    return Nlines

def drawLoad(xlim = (-4, 4), ylim = (-4, 4)):

    ax.cla()

    ax.set(xlim = xlim, ylim = ylim)
    ax.set_aspect('equal')

    ax.scatter(pnts[0, 0], pnts[1, 0], s = 20, marker = 'o', Color = 'red')
    ax.scatter(pnts[0, 1:], pnts[1, 1:], s = 20, marker = 'o', Color = 'gray')

    for i in range(Nlines):
        ax.plot([lines[2, i], lines[3, i]], [lines[0, i] * lines[2, i] + lines[1, i], lines[0, i] * lines[3, i] + lines[1, i]], linewidth = 3)
    
    fig.canvas.draw_idle()

def nextPnts(event):
    with mutex:
        firstPnt(pnts)

        t0 = time.time()
        for i in range(10):
            createPnts(pnts, N, shape = shape, mess = mess)
            getLines(lines, pnts, N, tol)
        print((time.time() - t0) / 10.0)

        drawLoad()

def updatePnts(val):
    global mess
    with mutex:
        mess = val
        createPnts(pnts, N, shape = shape, mess = mess)
        getLines(lines, pnts, N, tol)
        drawLoad()

def updateLinesTolerance(val):
    global tol
    
    with mutex:
        tol = val
        getLines(lines, pnts, N, tol)
    
    drawLoad(ax.get_xlim(), ax.get_ylim())

def updatePntsShape(event):
    global shape
    with mutex:
        shape += 1
        if shape > 1:
            shape = 0
        createPnts(pnts, N, shape = shape, mess = mess)
        getLines(lines, pnts, N, tol)
        drawLoad()

jit = False
def jitter(event):
    global jit
    
    def foo():
        while jit and plt.get_fignums():
            with mutex:
                pnts[:2, :] = pntsBuf + 0.02 * random.rand(2, N) - 0.01
                getLines(lines, pnts, N, tol)
                drawLoad(ax.get_xlim(), ax.get_ylim())
            time.sleep(0.5)

    with mutex:
        jit = not jit
        if jit:
            pntsBuf = pnts[:2, :]
        threading.Thread(target=foo).start()

def main():

    firstPnt(pnts)
    createPnts(pnts, N, shape = shape, mess = mess)

    getLines(lines, pnts, N, tol)
    drawLoad()

    ax1 = plt.axes([0.15, 0.17, 0.45, 0.03])
    ax2 = plt.axes([0.15, 0.14, 0.45, 0.03])
    ax3 = plt.axes([0.55, 0.28, 0.1, 0.04])
    ax4 = plt.axes([0.55, 0.35, 0.1, 0.04])
    ax5 = plt.axes([0.55, 0.42, 0.1, 0.04])

    sz1 = Slider(ax1, 'tolerance', 0.0, 0.8, tol, valstep = 0.02)
    sz1.on_changed(updateLinesTolerance)

    sz2 = Slider(ax2, 'mess', 0.0, 1.0, mess, valstep = 0.02)
    sz2.on_changed(updatePnts)

    btn1 = Button(ax3, 'Jitter', hovercolor='0.975')
    btn1.on_clicked(jitter)

    btn2 = Button(ax4, 'Shape', hovercolor='0.975')
    btn2.on_clicked(updatePntsShape)

    btn3 = Button(ax5, 'Next', hovercolor='0.975')
    btn3.on_clicked(nextPnts)

    plt.show()

if __name__ == "__main__":
    main()
