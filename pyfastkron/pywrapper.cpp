#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fastkron.h"

namespace py = pybind11;

#define THROW_ERROR(err) {\
  if ((err) != fastKronSuccess) {\
    throw std::runtime_error(fastKronGetErrorString(err));\
  }\
}

PYBIND11_MODULE(PyFastKronWrapper, m) {
  m.doc() = "..."; // optional module docstring

  py::enum_<fastKronBackend>(m, "Backend")
    .value("X86", fastKronBackend_X86)
    .value("ARM", fastKronBackend_ARM)
    .value("CUDA", fastKronBackend_CUDA)
    .value("HIP", fastKronBackend_HIP)
    .export_values();
  
  py::enum_<fastKronOp>(m, "Op")
    .value("N", fastKronOp_N)
    .value("T", fastKronOp_T)
    .export_values();

  m.def("init", [](uint32_t backends) {
    fastKronHandle handle;

    auto err = fastKronInit(&handle, backends);
    THROW_ERROR(err);

    return handle;
  }, "init");

  m.def("setOptions", [](fastKronHandle h, uint32_t options) {
    auto err = fastKronSetOptions(h, options);
    THROW_ERROR(err);
  }),

  m.def("destroy", fastKronDestroy, "destroy");

  m.def("initCUDA", [](fastKronHandle h, std::vector<uint32_t> ptrToStream, uint32_t gpus, uint32_t gpusInM, uint32_t gpusInK, uint32_t gpuLocalKrons) {
    auto err = fastKronInitCUDA(h, (void*)ptrToStream.data(), gpus, gpusInM, gpusInK, gpuLocalKrons);
    THROW_ERROR(err);
  }, "initCUDA");

  m.def("initHIP", [](fastKronHandle h, std::vector<uint32_t> ptrToStream) {
    auto err = fastKronInitHIP(h, (void*)ptrToStream.data());
    THROW_ERROR(err);
  }, "initHIP");

  m.def("initX86", [](fastKronHandle h) {
    auto err = fastKronInitX86(h);
    THROW_ERROR(err);
  }, "initX86");

  m.def("gekmmSizes", [](fastKronHandle handle, uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs) {
    size_t resultSize, tempSize;
    auto err = gekmmSizes(handle, M, N, Ps.data(), Qs.data(), &resultSize, &tempSize);
    THROW_ERROR(err);
    return py::make_tuple(resultSize, tempSize);
  }, "gekmmSizes");

  m.def("sgekmm", [](fastKronHandle handle, fastKronBackend backend, 
                     uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs,
                     uint64_t X, fastKronOp opX,
                     std::vector<uint64_t> Fs, fastKronOp opFs,
                     uint64_t Y, float alpha, float beta,
                     uint64_t Z, uint64_t temp1, uint64_t temp2) {
    auto err = sgekmm(handle, backend, M, N, Ps.data(), Qs.data(), (float*)X, opX, (float**)Fs.data(), opFs, (float*)Y, alpha, beta, (float*)Z, (float*)temp1, (float*)temp2);
    THROW_ERROR(err);
  }, "sgekmm");

  m.def("igekmm", [](fastKronHandle handle, fastKronBackend backend, 
                     uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs,
                     uint64_t X, fastKronOp opX,
                     std::vector<uint64_t> Fs, fastKronOp opFs,
                     uint64_t Y, int alpha, int beta,
                     uint64_t Z, uint64_t temp1, uint64_t temp2) {
    auto err = igekmm(handle, backend, M, N, Ps.data(), Qs.data(), (int*)X, opX, (int**)Fs.data(), opFs, (int*)Y, alpha, beta, (int*)Z, (int*)temp1, (int*)temp2);
    THROW_ERROR(err);
  }, "igekmm");

  m.def("dgekmm", [](fastKronHandle handle, fastKronBackend backend, 
                     uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs,
                     uint64_t X, fastKronOp opX,
                     std::vector<uint64_t> Fs, fastKronOp opFs,
                     uint64_t Y, double alpha, double beta,
                     uint64_t Z, uint64_t temp1, uint64_t temp2) {
    auto err = dgekmm(handle, backend, M, N, Ps.data(), Qs.data(), (double*)X, opX, (double**)Fs.data(), opFs, (double*)Y, alpha, beta, (double*)Z, (double*)temp1, (double*)temp2);
    THROW_ERROR(err);
  }, "dgekmm");
}