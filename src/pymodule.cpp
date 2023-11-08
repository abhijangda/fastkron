#include <Python.h>

#include "fastkron.h"

static PyObject* pyFastKronInit(PyObject* self, PyObject* args) {
  fastKronHandle handle;
  if (fastKronInit(&handle) != cudaSuccess)
    return Py_None;
  return PyLong_FromUnsignedLong((unsigned long)handle);
}

static PyObject* pyFastKronDestroy(PyObject* self, PyObject* args) {
  printf("Hello World\n");
  return Py_None;
}

static PyObject* pyKronGeMMSizes(PyObject* self, PyObject* args) {
  printf("Hello World\n");
  return Py_None;
}

static PyObject* pyKronSGEMM(PyObject* self, PyObject* args) {
  printf("Hello World\n");
  return Py_None;
}

// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
  {"pyFastKronInit",    pyFastKronInit,    METH_NOARGS, ""},
  {"pyFastKronDestroy", pyFastKronDestroy, METH_VARARGS, ""},
  {"pyKronGeMMSizes",   pyKronGeMMSizes,   METH_VARARGS, ""},
  {"pyKronSGEMM",       pyKronSGEMM,       METH_VARARGS, ""},
  {NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef fastKronModule = {
    PyModuleDef_HEAD_INIT,
    "FastKronCPP",
    "Python interface of FastKron",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_fastkroncpp(void) {
    return PyModule_Create(&fastKronModule);
}