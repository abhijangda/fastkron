#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fastkron.h"

namespace py = pybind11;

#define THROW_ERROR(err) {\
  if ((err) != fastKronSuccess) {\
    throw std::runtime_error(std::string("FastKronError: ") + std::string(fastKronGetErrorString(err)));\
  }\
}

#if defined(ENABLE_X86) && defined(ENABLE_CUDA)
PYBIND11_MODULE(FastKron, m)
#elif defined(ENABLE_X86)
PYBIND11_MODULE(FastKronX86, m)
#elif defined(ENABLE_CUDA)
PYBIND11_MODULE(FastKronCUDA, m)
#else
PYBIND11_MODULE(FastKron, m)
#endif

{
  m.doc() = "Python wrapper for FastKron C++ API. For more information on each function refers to C++ API.";

  py::enum_<fastKronBackend>(m, "Backend", py::module_local())
  
#if defined(ENABLE_X86) && defined(ENABLE_CUDA)
    .value("X86", fastKronBackend_X86)
    .value("ARM", fastKronBackend_ARM)
    .value("CUDA", fastKronBackend_CUDA)
    .value("HIP", fastKronBackend_HIP)
#elif defined(ENABLE_X86)
    .value("X86", fastKronBackend_X86)
#elif defined(ENABLE_CUDA)
    .value("CUDA", fastKronBackend_CUDA)
#endif
    .export_values();
  
  py::enum_<fastKronOp>(m, "Op", py::module_local())
    .value("N", fastKronOp_N)
    .value("T", fastKronOp_T)
    .export_values();

  m.def("init", []() {
    fastKronHandle handle;

    auto err = fastKronInitAllBackends(&handle);
    THROW_ERROR(err);

    return handle;
  }, "Initialize and return FastKron handle for all supported backends.");

  m.def("backends", []() {
    return fastKronGetBackends();
  }, "Return a bitwise OR of all supported FastKron.Backends.");

  m.def("version", []() {
    return fastKronVersion();
  }, "Return FastKron version.");

  m.def("setOptions", [](fastKronHandle h, uint32_t options) {
    auto err = fastKronSetOptions(h, options);
    THROW_ERROR(err);
  }, "Set options for a FastKron handle."),

  m.def("initCUDA", [](fastKronHandle h, std::vector<long> ptrToStream) {
    auto err = fastKronInitCUDA(h, (void*)ptrToStream.data());
    THROW_ERROR(err);
  }, "Initializes the CUDA backend with stream only if fastKronHandle was initialized with CUDA backend.");

  m.def("initHIP", [](fastKronHandle h, std::vector<long> ptrToStream) {
    auto err = fastKronInitHIP(h, (void*)ptrToStream.data());
    THROW_ERROR(err);
  }, "Initializes the HIP backend with stream only if fastKronHandle was initialized with HIP backend.");

  m.def("initX86", [](fastKronHandle h) {
    auto err = fastKronInitX86(h);
    THROW_ERROR(err);
  }, "Initializes the x86 backend with stream only if fastKronHandle was initialized with x86 backend.");

  m.def("gekmmSizes", [](fastKronHandle handle, uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs) {
    size_t resultSize, tempSize;
    auto err = gekmmSizes(handle, M, N, Ps.data(), Qs.data(), &resultSize, &tempSize);
    THROW_ERROR(err);
    return py::make_tuple(resultSize, tempSize);
  }, "Returns a tuple of number of elements of the result matrix and temporary matrices for GeKMM.");

  m.def("sgemkm", [](fastKronHandle handle, fastKronBackend backend, 
                     uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs,
                     uint64_t X, fastKronOp opX,
                     std::vector<uint64_t> Fs, fastKronOp opFs,
                     uint64_t Y, float alpha, float beta,
                     uint64_t Z, uint64_t temp1, uint64_t temp2) {
    auto err = sgemkm(handle, backend, M, N, Ps.data(), Qs.data(), (const float*)X, opX, (const float**)Fs.data(), opFs, (float*)Y, alpha, beta, (float*)Z, (float*)temp1, (float*)temp2);
    THROW_ERROR(err);
  }, "Perform GeKMM on using 32-bit floating point operations on input matrices.");

  m.def("dgemkm", [](fastKronHandle handle, fastKronBackend backend, 
                     uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs,
                     uint64_t X, fastKronOp opX,
                     std::vector<uint64_t> Fs, fastKronOp opFs,
                     uint64_t Y, double alpha, double beta,
                     uint64_t Z, uint64_t temp1, uint64_t temp2) {
    auto err = dgemkm(handle, backend, M, N, Ps.data(), Qs.data(), (const double*)X, opX, (const double**)Fs.data(), opFs, (double*)Y, alpha, beta, (double*)Z, (double*)temp1, (double*)temp2);
    THROW_ERROR(err);
  }, "Perform GeKMM on using 64-bit double floating point operations on input matrices");

  m.def("sgekmm", [](fastKronHandle handle, fastKronBackend backend, 
                     uint32_t N, std::vector<uint32_t> Qs, std::vector<uint32_t> Ps,
                     uint32_t M, 
                     std::vector<uint64_t> Fs, fastKronOp opFs,
                     uint64_t X, fastKronOp opX,
                     uint64_t Y, float alpha, float beta,
                     uint64_t Z, uint64_t temp1, uint64_t temp2) {
    auto err = sgekmm(handle, backend, N, Qs.data(), Ps.data(), M, (const float**)Fs.data(), opFs, (const float*)X, opX, (float*)Y, alpha, beta, (float*)Z, (float*)temp1, (float*)temp2);
    THROW_ERROR(err);
  }, "Perform GeKMM on using 32-bit floating point operations on input matrices.");

  m.def("dgekmm", [](fastKronHandle handle, fastKronBackend backend, 
                     uint32_t N, std::vector<uint32_t> Qs, std::vector<uint32_t> Ps,
                     uint32_t M,
                     std::vector<uint64_t> Fs, fastKronOp opFs,
                     uint64_t X, fastKronOp opX,
                     uint64_t Y, double alpha, double beta,
                     uint64_t Z, uint64_t temp1, uint64_t temp2) {
    auto err = dgekmm(handle, backend, N, Qs.data(), Ps.data(), M, (const double**)Fs.data(), opFs, (const double*)X, opX, (double*)Y, alpha, beta, (double*)Z, (double*)temp1, (double*)temp2);
    THROW_ERROR(err);
  }, "Perform GeKMM on using 64-bit double floating point operations on input matrices");

  m.def("sgemkmStridedBatched", [](fastKronHandle handle, fastKronBackend backend, 
                                    uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs,
                                    uint64_t X, fastKronOp opX, uint64_t strideX,
                                    std::vector<uint64_t> Fs, fastKronOp opFs, std::vector<uint64_t> strideF,
                                    uint64_t Y, uint64_t strideY, float alpha, float beta,
                                    uint32_t batchCount, uint64_t Z, uint64_t strideZ,
                                    uint64_t temp1, uint64_t temp2) {
    auto err = sgemkmStridedBatched(handle, backend, M, N, Ps.data(), Qs.data(), (const float*)X, opX, strideX,
                                    (const float**)Fs.data(), opFs, strideF.data(), (float*)Y, strideY, 
                                    alpha, beta, batchCount, (float*)Z, strideZ,
                                    (float*)temp1, (float*)temp2);
    THROW_ERROR(err);
  }, "Perform GeKMM Strided Batched on using 32-bit floating point operations on input matrices.");

  //TODO: make the order of arguments same as in cublas API m,n,k,alpha,x,f,z,beta,y
  m.def("dgemkmStridedBatched", [](fastKronHandle handle, fastKronBackend backend, 
                                    uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs,
                                    uint64_t X, fastKronOp opX, uint64_t strideX,
                                    std::vector<uint64_t> Fs, fastKronOp opFs, std::vector<uint64_t> strideF,
                                    uint64_t Y, uint64_t strideY, double alpha, double beta,
                                    uint32_t batchCount, uint64_t Z, uint64_t strideZ,
                                    uint64_t temp1, uint64_t temp2) {
    auto err = dgemkmStridedBatched(handle, backend, M, N, Ps.data(), Qs.data(), (const double*)X, opX, strideX,
                                    (const double**)Fs.data(), opFs, strideF.data(), (double*)Y, strideY, 
                                    alpha, beta, batchCount, (double*)Z, strideZ,
                                    (double*)temp1, (double*)temp2);
    THROW_ERROR(err);
  }, "Perform GeKMM Strided Batched on using 32-bit integer point operations on input matrices.");

  m.def("sgekmmStridedBatched", [](fastKronHandle handle, fastKronBackend backend, 
                                    uint32_t N, std::vector<uint32_t> Qs, std::vector<uint32_t> Ps,
                                    uint32_t M,
                                    std::vector<uint64_t> Fs, fastKronOp opFs, std::vector<uint64_t> strideF,
                                    uint64_t X, fastKronOp opX, uint64_t strideX,
                                    uint64_t Y, uint64_t strideY, float alpha, float beta,
                                    uint32_t batchCount, uint64_t Z, uint64_t strideZ,
                                    uint64_t temp1, uint64_t temp2) {
    auto err = sgekmmStridedBatched(handle, backend, N, Qs.data(), Ps.data(), M, 
                                    (const float**)Fs.data(), opFs, strideF.data(),
                                    (const float*)X, opX, strideX,
                                    (float*)Y, strideY, 
                                    alpha, beta, batchCount, (float*)Z, strideZ,
                                    (float*)temp1, (float*)temp2);
    THROW_ERROR(err);
  }, "Perform GeKMM Strided Batched on using 32-bit floating point operations on input matrices.");

  //TODO: make the order of arguments same as in cublas API m,n,k,alpha,x,f,z,beta,y
  m.def("dgekmmStridedBatched", [](fastKronHandle handle, fastKronBackend backend, 
                                    uint32_t N, std::vector<uint32_t> Qs, std::vector<uint32_t> Ps,
                                    uint32_t M,
                                    std::vector<uint64_t> Fs, fastKronOp opFs, std::vector<uint64_t> strideF,
                                    uint64_t X, fastKronOp opX, uint64_t strideX,
                                    uint64_t Y, uint64_t strideY, double alpha, double beta,
                                    uint32_t batchCount, uint64_t Z, uint64_t strideZ,
                                    uint64_t temp1, uint64_t temp2) {
    auto err = dgekmmStridedBatched(handle, backend, N, Qs.data(), Ps.data(), M,
                                    (const double**)Fs.data(), opFs, strideF.data(), 
                                    (const double*)X, opX, strideX,
                                    (double*)Y, strideY, 
                                    alpha, beta, batchCount, (double*)Z, strideZ,
                                    (double*)temp1, (double*)temp2);
    THROW_ERROR(err);
  }, "Perform GeKMM Strided Batched on using 32-bit integer point operations on input matrices.");

  //Forward functions
  m.def("smkmForward", [](fastKronHandle handle, fastKronBackend backend, 
                     uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs,
                     uint64_t X, fastKronOp opX,
                     std::vector<uint64_t> Fs, fastKronOp opFs,
                     uint64_t Y, std::vector<uint64_t> Intermediates) {
    auto err = smkmForward(handle, backend, M, N, Ps.data(), Qs.data(), (const float*)X, opX, (const float**)Fs.data(), opFs, (float*)Y, (float**)Intermediates.data());
    THROW_ERROR(err);
  }, "Perform GeKMM on using 32-bit floating point operations on input matrices.");

  m.def("dmkmForward", [](fastKronHandle handle, fastKronBackend backend, 
                     uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs,
                     uint64_t X, fastKronOp opX,
                     std::vector<uint64_t> Fs, fastKronOp opFs,
                     uint64_t Y, std::vector<uint64_t> Intermediates) {
    auto err = dmkmForward(handle, backend, M, N, Ps.data(), Qs.data(), (const double*)X, opX, (const double**)Fs.data(), opFs, (double*)Y, (double**)Intermediates.data());
    THROW_ERROR(err);
  }, "Perform GeKMM on using 64-bit double floating point operations on input matrices");

  m.def("gekmmSizesForward", [](fastKronHandle handle, uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs) {
    size_t resultSize;
    std::vector<size_t> intermediateSizes(std::max(N-1, 1U), 0);
    auto err = gekmmSizesForward(handle, M, N, Ps.data(), Qs.data(), &resultSize, intermediateSizes.data());
    THROW_ERROR(err);
    return py::make_tuple(resultSize, intermediateSizes);
  }, "Returns a tuple of number of elements of the result matrix and temporary matrices for GeKMM.");

  m.def("skmmForward", [](fastKronHandle handle, fastKronBackend backend, 
                     uint32_t N, std::vector<uint32_t> Qs, std::vector<uint32_t> Ps,
                     uint32_t M, 
                     std::vector<uint64_t> Fs, fastKronOp opFs,
                     uint64_t X, fastKronOp opX,
                     uint64_t Y, std::vector<uint64_t> Intermediates) {
    auto err = skmmForward(handle, backend, N, Qs.data(), Ps.data(), M, (const float**)Fs.data(), opFs, (const float*)X, opX, (float*)Y, (float**)Intermediates.data());
    THROW_ERROR(err);
  }, "Perform GeKMM on using 32-bit floating point operations on input matrices.");

  m.def("dkmmForward", [](fastKronHandle handle, fastKronBackend backend, 
                     uint32_t N, std::vector<uint32_t> Qs, std::vector<uint32_t> Ps,
                     uint32_t M,
                     std::vector<uint64_t> Fs, fastKronOp opFs,
                     uint64_t X, fastKronOp opX,
                     uint64_t Y, std::vector<uint64_t> Intermediates) {
    auto err = dkmmForward(handle, backend, N, Qs.data(), Ps.data(), M, (const double**)Fs.data(), opFs, (const double*)X, opX, (double*)Y, (double**)Intermediates.data());
    THROW_ERROR(err);
  }, "Perform GeKMM on using 64-bit double floating point operations on input matrices");

  m.def("smkmForwardStridedBatched", [](fastKronHandle handle, fastKronBackend backend, 
                                    uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs,
                                    uint64_t X, fastKronOp opX, uint64_t strideX,
                                    std::vector<uint64_t> Fs, fastKronOp opFs, std::vector<uint64_t> strideF,
                                    uint64_t Y, uint64_t strideY, uint32_t batchCount,
                                    std::vector<uint64_t> Intermediates, std::vector<uint64_t> strideIntermediates) {
    auto err = smkmForwardStridedBatched(handle, backend, M, N, Ps.data(), Qs.data(), (const float*)X, opX, strideX,
                                    (const float**)Fs.data(), opFs, strideF.data(), (float*)Y, strideY, 
                                    batchCount, (float**)Intermediates.data(), strideIntermediates.data());
    THROW_ERROR(err);
  }, "Perform GeKMM Strided Batched on using 32-bit floating point operations on input matrices.");

  //TODO: make the order of arguments same as in cublas API m,n,k,alpha,x,f,z,beta,y
  m.def("dmkmForwardStridedBatched", [](fastKronHandle handle, fastKronBackend backend, 
                                    uint32_t M, uint32_t N, std::vector<uint32_t> Ps, std::vector<uint32_t> Qs,
                                    uint64_t X, fastKronOp opX, uint64_t strideX,
                                    std::vector<uint64_t> Fs, fastKronOp opFs, std::vector<uint64_t> strideF,
                                    uint64_t Y, uint64_t strideY,
                                    uint32_t batchCount, std::vector<uint64_t> Intermediates, std::vector<uint64_t> strideIntermediates) {
    auto err = dmkmForwardStridedBatched(handle, backend, M, N, Ps.data(), Qs.data(), (const double*)X, opX, strideX,
                                    (const double**)Fs.data(), opFs, strideF.data(), (double*)Y, strideY, 
                                    batchCount, (double**)Intermediates.data(), strideIntermediates.data());
    THROW_ERROR(err);
  }, "Perform GeKMM Strided Batched on using 32-bit integer point operations on input matrices.");

  m.def("skmmForwardStridedBatched", [](fastKronHandle handle, fastKronBackend backend, 
                                    uint32_t N, std::vector<uint32_t> Qs, std::vector<uint32_t> Ps,
                                    uint32_t M,
                                    std::vector<uint64_t> Fs, fastKronOp opFs, std::vector<uint64_t> strideF,
                                    uint64_t X, fastKronOp opX, uint64_t strideX,
                                    uint64_t Y, uint64_t strideY,
                                    uint32_t batchCount, std::vector<uint64_t> Intermediates, std::vector<uint64_t> strideIntermediates) {
    auto err = skmmForwardStridedBatched(handle, backend, N, Qs.data(), Ps.data(), M, 
                                    (const float**)Fs.data(), opFs, strideF.data(),
                                    (const float*)X, opX, strideX,
                                    (float*)Y, strideY, 
                                    batchCount, (float**)Intermediates.data(), strideIntermediates.data());
    THROW_ERROR(err);
  }, "Perform GeKMM Strided Batched on using 32-bit floating point operations on input matrices.");

  //TODO: make the order of arguments same as in cublas API m,n,k,alpha,x,f,z,beta,y
  m.def("dkmmForwardStridedBatched", [](fastKronHandle handle, fastKronBackend backend, 
                                    uint32_t N, std::vector<uint32_t> Qs, std::vector<uint32_t> Ps,
                                    uint32_t M,
                                    std::vector<uint64_t> Fs, fastKronOp opFs, std::vector<uint64_t> strideF,
                                    uint64_t X, fastKronOp opX, uint64_t strideX,
                                    uint64_t Y, uint64_t strideY,
                                    uint32_t batchCount, std::vector<uint64_t> Intermediates, std::vector<uint64_t> strideIntermediates) {
    auto err = dkmmForwardStridedBatched(handle, backend, N, Qs.data(), Ps.data(), M,
                                    (const double**)Fs.data(), opFs, strideF.data(), 
                                    (const double*)X, opX, strideX,
                                    (double*)Y, strideY, 
                                    batchCount,(double**)Intermediates.data(), strideIntermediates.data());
    THROW_ERROR(err);
  }, "Perform GeKMM Strided Batched on using 32-bit integer point operations on input matrices.");

  m.def("destroy", [](fastKronHandle handle) {
    fastKronDestroy(handle);
  }, "Destroy and deallocate a FastKron handle.");
}