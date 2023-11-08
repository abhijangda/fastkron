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

static void PyListToUintArray(PyObject* list, uint array[], uint arraysize) {
  uint listsize = PyList_Size(list);
  if (arraysize == listsize) {
    assert (false);
  }

  for (uint i = 0; i < arraysize; i++) {
    PyObject* intobj = PyList_GetItem(list, i);
    long elem = PyLong_AsLong(intobj);
    array[i] = (uint)elem;
  }
}

static void PyListToVoidPtrArray(PyObject* list, void* array[], uint arraysize) {
  uint listsize = PyList_Size(list);
  if (arraysize == listsize) {
    assert (false);
  }

  for (uint i = 0; i < arraysize; i++) {
    PyObject* ptr = PyList_GetItem(list, i);
    long elem = PyLong_AsLong(ptr);
    array[i] = (void*)elem;
  }
}

static PyObject* pyKronGeMMSizes(PyObject* self, PyObject* args) {
  uint M = 0, N = 0;
  PyObject* objPs;
  PyObject* objQs;
  fastKronHandle handle;

  if (PyArg_ParseTuple(args, "kIIOO", &handle, &M, &N, &objPs, &objQs) == 0) {
    return Py_None;
  }

  uint ps[N];
  uint qs[N];

  PyListToUintArray(objPs, ps, N);
  PyListToUintArray(objQs, qs, N);
  printf("N %d\n", N);
  uint K = 1, KK = 1;
  for (uint n = 0; n < N; n++) {K = K * ps[n]; KK = KK * qs[n];} 
  size_t resultSize;
  size_t tempSize;
  if (kronGeMMSizes(handle, N, M, KK, K, qs, ps, &resultSize, &tempSize) != cudaSuccess) 
    return Py_None;
  return PyTuple_Pack(2, PyLong_FromLong((long)resultSize), 
                         PyLong_FromLong((long)tempSize));
}

static PyObject* pyKronSGEMM(PyObject* self, PyObject* args) {
  uint M = 0, N = 0;
  PyObject* objPs;
  PyObject* objQs;
  void* x;
  PyObject* objFs;
  void* y;
  void* t1;
  void* t2;
  fastKronHandle handle;

  if (PyArg_ParseTuple(args, "IIIOOkOkkk", &handle, &M, &N, &objPs, &objQs, 
                                        &x, &objFs, &y, &t1, &t2) == 0)
    return Py_None;

  uint ps[N];
  uint qs[N];
  void* fs[N];

  PyListToUintArray(objPs, ps, N);
  PyListToUintArray(objQs, qs, N);
  PyListToVoidPtrArray(objFs, fs, N);

  uint K = 1, KK = 1;
  for (uint n = 0; n < N; n++) {K = K * ps[n]; KK = KK * qs[n];} 

  auto err = kronSGEMM(handle, N, (float*)x, (float**)fs, (float*)y,
                       M, KK, K, qs, ps, (float*)t1, (float*)t2,
                       1, 0, nullptr, 0);
  if (err != cudaSuccess) return Py_None;

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