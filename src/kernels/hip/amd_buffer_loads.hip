
using int32x4_t = __attribute__( (__vector_size__(4 * sizeof(int)) )) int;
using float32x4_t = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;

#if defined(__gfx803__) || defined(__gfx900__) || defined(__gfx906__) || defined(__gfx908__) || \
    defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) ||                          \
    defined(__gfx942__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#elif defined(__gfx1030__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31014000
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31004000
#else
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#endif


template <typename T>
union BufferResource {
  __device__ constexpr BufferResource() : content{} {}

  // 128 bit SGPRs to supply buffer resource in buffer instructions
  // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
  int32x4_t content;
  T* address[2];
  int range[4];
  int config[4];
};


template <typename T>
__device__ int32x4_t make_wave_buffer_resource(T* p_wave, int element_space_size) {
  BufferResource<T> wave_buffer_resource;

  // wavewise base address (64 bit)
  wave_buffer_resource.address[0] = const_cast<T*>(p_wave);
  // wavewise range (32 bit)
  wave_buffer_resource.range[2] = element_space_size * sizeof(T);
  // wavewise setting (32 bit)
  wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

  return wave_buffer_resource.content;
}

// buffer load i16
CUDA_DEVICE
ushort llvm_amdgcn_raw_buffer_load_16b(int32x4_t srsrc,
                                int voffset,
                                int soffset,
                                int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i16");
CUDA_DEVICE
float llvm_amdgcn_raw_buffer_load_32b(int32x4_t srsrc,
                                int voffset,
                                int soffset,
                                int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");
CUDA_DEVICE
float2 llvm_amdgcn_raw_buffer_load_32bx2(int32x4_t srsrc,
                                  int voffset,
                                  int soffset,
                                  int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f32");
CUDA_DEVICE
float32x4_t llvm_amdgcn_raw_buffer_load_32bx4(int32x4_t srsrc,
                                  int voffset,
                                  int soffset,
                                  int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f32");

enum struct AmdBufferCoherenceEnum {
    DefaultCoherence = 0, // default value
    GLC              = 1,
    SLC              = 2,
    GLC_SLC          = 3,
};

template <typename ElemT, typename VecT, AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__
void amd_buffer_load(const ElemT* p_src_wave, int src_thread_element_offset,
                     int src_element_space_size, ElemT regs[]) {
  const int32x4_t src_wave_buffer_resource =
      make_wave_buffer_resource(p_src_wave, src_element_space_size);

  int src_thread_addr_offset = src_thread_element_offset * sizeof(ElemT);
  static_assert(sizeof(ElemT) == 4,
                "wrong! not implemented");

  switch (sizeof(ElemT)) {
    case 4: switch (sizeof(VecT)/sizeof(ElemT)) {
      // case 1: {
      //   auto tmp = llvm_amdgcn_raw_buffer_load_32b(src_wave_buffer_resource,
      //                                              src_thread_addr_offset,
      //                                              src_wave_addr_offset,
      //                                              static_cast<int>(coherence));
      //   regs[0] = tmp;
      //   return;
      // }
      // case 2: {
      //   auto tmp = llvm_amdgcn_raw_buffer_load_32bx2(src_wave_buffer_resource,
      //                                                src_thread_addr_offset,
      //                                                src_wave_addr_offset,
      //                                                static_cast<int>(coherence));
      //   regs[0] = tmp.x; regs[1] = tmp.y;
      //   return;
      // }
      case 4: {
        auto tmp = llvm_amdgcn_raw_buffer_load_32bx4(src_wave_buffer_resource,
                                                     src_thread_addr_offset,
                                                     0,
                                                     static_cast<int>(coherence));
        regs[0] = tmp[0]; regs[1] = tmp[1]; regs[2] = tmp[2]; regs[3] = tmp[3];
        return;
      }
    }
  }

  return;
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
// each thread then writes to p_dest_wave, local registers
// template <typename ElemT, typename VecT,
//           AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
// CUDA_DEVICE 
// void amd_buffer_load(const ElemT* p_src_wave, int src_thread_element_offset,
//                      int src_element_space_size, ElemT p_dest_wave[]) {
//   const int32x4_t src_wave_buffer_resource =
//       make_wave_buffer_resource(p_src_wave, src_element_space_size);

//   int src_thread_addr_offset = src_thread_element_offset * sizeof(ElemT);
//   amd_buffer_load_impl_raw<ElemT, sizeof(ElemT), sizeof(VecT)/sizeof(ElemT)>(src_wave_buffer_resource,
//                                                                              src_thread_addr_offset, 0,
//                                                                              p_dest_wave);
// }