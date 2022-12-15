#include "Python.h"
#include "numpy/arrayobject.h"

#pragma warning(disable:4996)		//unsafe functions
#pragma warning(disable:4244)		//неявное из float в int
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

class CPy {

public:
    double* XY;
    int N;

    CPy(double a, int N_ = 1) : N(N_), XY(new double[N_]) {
        if (a > 2.0)
            delete this;
            return;
        for (int i = 0; i < N; i++)
            XY[i] = a;
    }

};

static vector<CPy*> vec;

PyObject* init(PyObject*, PyObject* o) {
    double val = PyFloat_AsDouble(PyTuple_GetItem(o, 0));
    vec.push_back(new CPy(val));
    cout << "ID: " << vec.size() - 1 << endl;
    return PyLong_FromLong(vec.size() - 1);
}

PyObject* Foo(PyObject*, PyObject* o) {
    return PyFloat_FromDouble(vec[PyLong_AsLong(PyTuple_GetItem(o, 0))]->XY[0]);
}

static PyMethodDef Tempo_methods[] = {
    { "init", (PyCFunction)init, METH_VARARGS, "_"},
    { "Foo", (PyCFunction)Foo, METH_VARARGS, "_"},
    { nullptr, nullptr, 0, nullptr }
};

static PyModuleDef Tempo = {
    PyModuleDef_HEAD_INIT,
    "Tempo",                        // Module name to use with Python import statements
    "Just training",  // Module description
    0,
    Tempo_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_Tempo() {
    return PyModule_Create(&Tempo);
}