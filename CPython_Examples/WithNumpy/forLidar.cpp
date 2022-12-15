#include "Python.h"
#include "numpy/arrayobject.h"

#pragma warning(disable:4996)		//unsafe functions
#pragma warning(disable:4244)		//неявное из float в int
#include <iostream>
#include <cmath>

using namespace std;

PyObject* return_0 = PyLong_FromLong(0);
PyObject* return_1 = PyLong_FromLong(1);

int N = 0;  //num of points
double** xy = nullptr;
double** dxy = nullptr;
double** pins = nullptr;
//double** pins0 = nullptr;
//double** pins1 = nullptr;
double** z = nullptr;
int* actualPos = nullptr;

int M = 0;
int edgeShift = 0;
int deep = 0;
int lever = 0;
double dist0 = 0.0;

int NofPins = 0;
int NofPairs = 0;

const int p = 2; //different step

PyObject* py_pins = nullptr;
PyArrayObject* py_pins_pins = nullptr;
PyArrayObject* py_pins_pins0 = nullptr;
PyArrayObject* py_pins_pins1 = nullptr;
PyArrayObject* py_xy = nullptr;
PyArrayObject* py_dxy = nullptr;

PyObject* forLidar(PyObject*, PyObject* o) {
    if (PyTuple_GET_SIZE(o) == 3) {

        py_pins = PyTuple_GetItem(o, 0);
        py_xy = (PyArrayObject*)PyTuple_GetItem(o, 1);
        py_dxy = (PyArrayObject*)PyTuple_GetItem(o, 2);

        py_pins_pins = (PyArrayObject*)PyObject_GetAttr(py_pins, PyUnicode_FromString("pins"));
        py_pins_pins0 = (PyArrayObject*)PyObject_GetAttr(py_pins, PyUnicode_FromString("pins0"));
        py_pins_pins1 = (PyArrayObject*)PyObject_GetAttr(py_pins, PyUnicode_FromString("pins1"));

        N = (int)PyLong_AsLong(PyObject_GetAttr(py_pins, PyUnicode_FromString("N")));
        M = (int)PyLong_AsLong(PyObject_GetAttr(py_pins, PyUnicode_FromString("M")));
        edgeShift = (int)PyLong_AsLong(PyObject_GetAttr(py_pins, PyUnicode_FromString("edgeShift")));
        deep = (int)PyLong_AsLong(PyObject_GetAttr(py_pins, PyUnicode_FromString("deep")));
        lever = (int)PyLong_AsLong(PyObject_GetAttr(py_pins, PyUnicode_FromString("lever")));
        dist0 = (double)PyFloat_AsDouble(PyObject_GetAttr(py_pins, PyUnicode_FromString("dist0")));

        if (PyErr_Occurred()) {
            cerr << "Wrong attribute(s)" << endl;
            PyErr_Clear();
            return return_0;
        }

        if (PyArray_NDIM(py_pins_pins) != 2 &&
            PyArray_NDIM(py_pins_pins0) != 2 &&
            PyArray_NDIM(py_pins_pins1) != 2) {
            cerr << "Wrong pins attributes dimensions" << endl;
            return return_0;
        }

        if (PyArray_NDIM(py_xy) != 2 or PyArray_NDIM(py_dxy) != 2) {
            cerr << "Wrong xy or dxy" << endl;
            return return_0;
        }

        xy = new double* [2];     //here we work with simple coordinates, not homogeneous
        dxy = new double* [2];
        pins = new double* [5];
        //pins0 = new double* [2];
        //pins1 = new double* [2];
        z = new double* [2];
        for (int i = 0; i < 2; i++) {
            xy[i] = new double[N];
            dxy[i] = new double[N];
            //pins0[i] = new double[N];
            //pins1[i] = new double[N];
            z[i] = new double[N];
        }

        for (int i = 0; i < 5; i++)
            pins[i] = new double[N];

        actualPos = new int[N];
        return return_1;
    }
    else {
        cerr << "Incorrect args" << endl;
        return return_0;
    }
}

PyObject* ExtractPins(PyObject*, PyObject* o) {
    NofPins = 0;
    if (py_pins != nullptr) {
        int NArgs = PyTuple_GET_SIZE(o);
        if (NArgs == 0 || NArgs == 3) {

            if (NArgs == 3) {   //updating xy and dxy by Arg0 and Arg1
                PyArrayObject* py_xy_t = (PyArrayObject*)PyTuple_GetItem(o, 0);
                PyArrayObject* py_dxy_t = (PyArrayObject*)PyTuple_GetItem(o, 1);
                int N_t = PyLong_AsLong(PyTuple_GetItem(o, 2));

                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < N_t; j++) {
                        xy[i][j] = *(double*)PyArray_GETPTR2(py_xy_t, i, j); //simple coords, not homogeneous
                        dxy[i][j] = *(double*)PyArray_GETPTR2(py_dxy_t, i, j);
                        z[i][j] = 0.0;
                    }
            }

            int NN = 0;
            for (int k = edgeShift; k < N - edgeShift; k++) {
                if (dxy[0][k] || dxy[1][k]) {
                    actualPos[NN] = k;
                    NN++;
                }
            }

            for (int k = 0; k < NN - p - M + 1; k++) {
                int pos = actualPos[k + p];
                for (int i = k + p; i < k + p + M; i++) {
                    int n = actualPos[i];
                    int n_p = actualPos[i - p];
                    double alpha = atan2(dxy[0][n_p] * dxy[1][n] - dxy[1][n_p] * dxy[0][n], dxy[0][n_p] * dxy[0][n] + dxy[1][n_p] * dxy[1][n]);
                    double vn_pLen = sqrt(dxy[0][n_p] * dxy[0][n_p] + dxy[1][n_p] * dxy[1][n_p]);
                    double vnLen = sqrt(dxy[0][n] * dxy[0][n] + dxy[1][n] * dxy[1][n]);
                    if (abs(alpha) < 0.20943951) {  //12 degrees
                        z[0][pos] *= pow(2048.0, 1.0 / (M - 1.0));
                        z[1][pos] *= pow(2048.0, 1.0 / (M - 1.0));
                    }
                    else {
                        z[0][pos] = pow(2048.0, 1.0 / (M - 1.0)) * z[0][pos] + /*deltaX*/ vnLen * cos(alpha) - vn_pLen;
                        z[1][pos] = pow(2048.0, 1.0 / (M - 1.0)) * z[1][pos] + /*deltaY*/ vnLen * sin(alpha);
                    }
                }
            }

            int i = 2;
            while (i < NN - M + 1) {
                int pos = actualPos[i];

                if (sqrt(z[0][pos] * z[0][pos] + z[1][pos] * z[1][pos]) > dist0) {
                    bool acceptBwd = true;
                    bool acceptFwd = true;

                    for (int iBwd = pos - 1; iBwd >= pos - deep; iBwd--) {
                        if (!(z[0][iBwd] || z[1][iBwd])) {
                            acceptBwd = false;
                            break;
                        }
                        if (sqrt(z[0][iBwd + 1] * z[0][iBwd + 1] + z[1][iBwd + 1] * z[1][iBwd + 1]) < sqrt(z[0][iBwd] * z[0][iBwd] + z[1][iBwd] * z[1][iBwd])) {
                            acceptBwd = false;
                            break;
                        }
                    }

                //Seems better without checking forward
                    //for (int iFwd = pos + 1; iFwd <= pos + deep; iFwd++) {
                    //    if (!(z[0][iFwd] || z[1][iFwd])) {
                    //        acceptFwd = false;
                    //        break;
                    //    }
                    //    if (sqrt(z[0][iFwd] * z[0][iFwd] + z[1][iFwd] * z[1][iFwd]) > sqrt(z[0][iFwd - 1] * z[0][iFwd - 1] + z[1][iFwd - 1] * z[1][iFwd - 1])) {
                    //        acceptFwd = false;
                    //        break;
                    //    }
                    //}

                    if (acceptBwd || acceptFwd) {
                        pins[0][NofPins] = xy[0][pos];
                        pins[1][NofPins] = xy[1][pos];
                        pins[2][NofPins] = (double)pos;
                        pins[3][NofPins] = z[0][pos];
                        pins[4][NofPins] = z[1][pos];
                        i += 3;
                        NofPins++;
                        continue;
                    }
                }

                i++;
            }

            //write pins.pins
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < NofPins; j++)
                    *(double*)PyArray_GETPTR2(py_pins_pins, i, j) = pins[i][j];

        }
        else
            cerr << "Invalid args" << endl;
    }
    else
        cerr << "Not initialized" << endl;

    PyObject_SetAttr(py_pins, PyUnicode_FromString("NofPins"), PyLong_FromLong(NofPins));
    return PyLong_FromLong(NofPins);
}

PyObject* FindPinPairs(PyObject*, PyObject* o) {

    NofPairs = 0;
    if (py_pins != nullptr) {
        int NArgs = PyTuple_GET_SIZE(o);
        if (NArgs == 0 || NArgs == 1) {

            N = (int)PyLong_AsLong(PyObject_GetAttr(py_pins, PyUnicode_FromString("N")));

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < N; j++) {
                    xy[i][j] = *(double*)PyArray_GETPTR2(py_xy, i, j); //simple coords, not homogeneous
                    dxy[i][j] = *(double*)PyArray_GETPTR2(py_dxy, i, j);
                    z[i][j] = 0.0;
                }

            if (NArgs == 1) {   //updating pins_pins by Arg0
                PyArrayObject* py_pins_pins_t = (PyArrayObject*)PyObject_GetAttr(PyTuple_GetItem(o, 0), PyUnicode_FromString("pins"));
                NofPins = PyLong_AsLong(PyObject_GetAttr(PyTuple_GetItem(o, 0), PyUnicode_FromString("NofPins")));

                for (int i = 0; i < 5; i++)
                    for (int j = 0; j < NofPins; j++)
                        pins[i][j] = *(double*)PyArray_GETPTR2(py_pins_pins_t, i, j);
            }

            if (NofPins > 2) {

                for (int pinNumber = 0; pinNumber < NofPins; pinNumber++) { //for pin number...
                    int basePos = (int)pins[2][pinNumber];

                    //clarifying zero position
                    int lBasePos = basePos - lever;
                    if (lBasePos < 0)
                        lBasePos = 0;

                    int rBasePos = basePos + lever;
                    if (rBasePos > N)
                        rBasePos = N;

                    double phi = atan2(pins[1][pinNumber], pins[0][pinNumber]);
                    double phiDist = HUGE_VAL;
                    for (int i = lBasePos; i < rBasePos; i++) {
                        if (abs(atan2(xy[1][i], xy[0][i]) - phi) < phiDist) {
                            phiDist = abs(atan2(xy[1][i], xy[0][i]) - phi);
                            basePos = i;
                        }
                    }

                    int n = 0;
                    int posi = 0;
                    while (basePos >= 0) {
                        if (dxy[0][basePos] || dxy[1][basePos]) {
                            posi = basePos;
                            n++;
                            if (n == lever + p + 1)
                                break;
                        }
                        basePos--;
                    }
                    
                    int NN = 0;
                    while (posi < N) {
                        if (dxy[0][posi] || dxy[1][posi]) {
                            actualPos[NN] = posi;
                            NN++;
                            if (NN == n + lever + M) //maybe it should be "if (n_ == n + lever + M - 1)""
                                break;
                        }
                        posi++;
                    }

                    int corrPinPos = -1;
                    double dist;
                    double dist0_ = dist0;
                    for (int k = 0; k < NN - p - M + 1; k++) {  //for point number...
                        int pos = actualPos[k + p];
                        double zX = 0.0;
                        double zY = 0.0;
                        for (int j = k + p; j < k + p + M; j++) {
                            int n = actualPos[j];
                            int n_p = actualPos[j - p];
                            double alpha = atan2(dxy[0][n_p] * dxy[1][n] - dxy[1][n_p] * dxy[0][n], dxy[0][n_p] * dxy[0][n] + dxy[1][n_p] * dxy[1][n]);
                            double vn_pLen = sqrt(dxy[0][n_p] * dxy[0][n_p] + dxy[1][n_p] * dxy[1][n_p]);
                            double vnLen = sqrt(dxy[0][n] * dxy[0][n] + dxy[1][n] * dxy[1][n]);
                            if (abs(alpha) < 0.20943951) {  //12 degrees
                                zX *= pow(2048.0, 1.0 / (M - 1.0));
                                zY *= pow(2048.0, 1.0 / (M - 1.0));
                            }
                            else {
                                zX = pow(2048.0, 1.0 / (M - 1.0)) * zX + /*deltaX*/ vnLen * cos(alpha) - vn_pLen;
                                zY = pow(2048.0, 1.0 / (M - 1.0)) * zY + /*deltaY*/vnLen * sin(alpha);
                            }
                        }

                        dist = sqrt((zX - pins[3][pinNumber]) * (zX - pins[3][pinNumber]) +
                            (zY - pins[4][pinNumber]) * (zY - pins[4][pinNumber]));

                        if (dist < dist0_) {
                            dist0_ = dist;
                            corrPinPos = pos;
                        }
                    }

                    if (corrPinPos >= 0) {
                        *(double*)PyArray_GETPTR2(py_pins_pins0, 0, NofPairs) = pins[0][pinNumber];
                        *(double*)PyArray_GETPTR2(py_pins_pins0, 1, NofPairs) = pins[1][pinNumber];
                        *(double*)PyArray_GETPTR2(py_pins_pins1, 0, NofPairs) = xy[0][corrPinPos];
                        *(double*)PyArray_GETPTR2(py_pins_pins1, 1, NofPairs) = xy[1][corrPinPos];
                        //pins0[0][NofPairs] = pins[0][pinNumber];
                        //pins0[1][NofPairs] = pins[1][pinNumber];
                        //pins1[0][NofPairs] = xy[0][corrPinPos];
                        //pins1[1][NofPairs] = xy[1][corrPinPos];
                        NofPairs++;
                    }
                }
            }
        }
        else
            cerr << "Incorrect args" << endl;
    }
    else
        cerr << "Not initialized" << endl;
   
    PyObject_SetAttr(py_pins, PyUnicode_FromString("NofPairs"), PyLong_FromLong(NofPairs));
    return PyLong_FromLong(NofPairs);
}

static PyMethodDef forLidar_methods[] = {
    { "ExtractPins", (PyCFunction)ExtractPins, METH_VARARGS, "Be sure for call FindPinPairs() at first or place 3 args. Args: none or (xy, dxy, N) if special. Returns NofPins, writes pins.pins and pins.NofPins"},
    { "FindPinPairs", (PyCFunction)FindPinPairs, METH_VARARGS, "Args: none or (pins) if special. Returns NofPairs, writes pins.NofPairs, pins.pins0 and pins.pins1"},
    { "forLidar", (PyCFunction)forLidar, METH_VARARGS, "Initialize with pins main instance and xy, dxy"},
    { nullptr, nullptr, 0, nullptr }
};

static PyModuleDef forLidar_forLidar = {
    PyModuleDef_HEAD_INIT,
    "forLidar",                        // Module name to use with Python import statements
    "Lidar processing acceleration",  // Module description
    0,
    forLidar_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_forLidar() {
    return PyModule_Create(&forLidar_forLidar);
}

PyMODINIT_FUNC PyFini_forLidar() {
    return PyModule_Create(&forLidar_forLidar);
}