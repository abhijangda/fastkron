#include "anyoption.h"
#include "testBase.h"

#include <sstream>

bool parseStringToIntegers(char* str, uint array[], const int numInts) {
  int parsedInts = 0;
  
  std::string stdstr(str);
  std::stringstream stream(stdstr);
  uint n;
  char comma;
  while(stream >> n){
    array[parsedInts] = n;
    stream >> comma;
    parsedInts++;
    if (comma != ',')
      break;
  }
  if (parsedInts == 1) {
    for (; parsedInts < numInts; parsedInts++) {
      array[parsedInts] = array[0];
    }
  }

  if (parsedInts != numInts) return false;

  return true;
}

/**************************************************
              Main Function
***************************************************/
int main(int argc, char* argv[]) {  
  int rows = 0;
  int facs = 0;
  char* fac_rows = NULL;
  char* fac_cols = NULL;
  fastKronOp opf = fastKronOp_N;
  fastKronOp opx = fastKronOp_N;
  char* type = NULL;
  bool checkResults = false;
  int runs = 0;
  int warmup = 0;
  bool useUVA = false;
  int gpuLocalKrons = 0;
  int gpus = 1;
  int gpuInRows = 0;
  int gpuInCols = 0;
  bool multiGPU = false;
  bool useFusion = false;
  bool tune = false;
  fastKronBackend backend = fastKronBackend_NONE;

  AnyOption *opt = new AnyOption();

  opt->addUsage("Performs KronMatmul of matrix X[M, K] with Kronecker Product of N matrices of shape F[Pi, Qi]");
  opt->addUsage("rows: Number of Rows of X");
  opt->addUsage("facs:  Number of Kron Factors");
  opt->addUsage("fac_rows: Rows of each Kron Factor separated by comma");
  opt->addUsage("fac_cols: Cols of each Kron Factor separated by comma");
  opt->addUsage("type:  Type of matrices (float, double, int, long)");
  opt->addUsage("check: Check results for first run");
  opt->addUsage("runs:  Number of runs");
  opt->addUsage("warmup:  Number of warmup runs");
  opt->addUsage("uva: Allocate and run using NVIDIA UVA");
  opt->addUsage("backend: Backend one of CUDA, ROCM, X86, ARM");
  opt->addUsage("gpurows: Rows for temp on GPU. valid only with uva");
  opt->addUsage("maxkronbatch: Factors rows per inner iteration. valid only with uva");
  opt->addUsage("nummaxkronbatch");

  opt->setOption("rows", 'm');
  opt->setOption("facs", 'n');
  opt->setOption("opx");
  opt->setOption("opf");
  opt->setOption("fac_rows", 'p');
  opt->setOption("fac_cols", 'q');
  opt->setOption("type", 't');
  opt->setOption("runs", 'r');
  opt->setOption("warmup", 'w');
  
  opt->setFlag("check", 'c');
  opt->setFlag("uva", 'u');
  opt->setOption("gpuLocalKrons");
  opt->setOption("gpus");
  opt->setOption("GM");
  opt->setOption("GK");

  opt->setFlag("fuse");
  opt->setFlag("tune");

  opt->setOption("backend");

  opt->processCommandArgs(argc, argv);
  
  if (!opt->hasOptions()) { /* print usage if no options */
    opt->printUsage();
    delete opt;
    return 1;
  }

  if (opt->getValue("backend") == NULL) {
    std::cout << "No backend specific" << std::endl;
    return 1;
  } else {
    char* backendStr = opt->getValue("backend");
    if (strcmp(backendStr, "CUDA") == 0) {
      backend = fastKronBackend_CUDA;
    } else if (strcmp(backendStr, "ROCM") == 0) {
      backend = fastKronBackend_ROCM;
    } else if (strcmp(backendStr, "X86") == 0) {
      backend = fastKronBackend_X86;
    } else if (strcmp(backendStr, "ARM") == 0) {
      backend = fastKronBackend_ARM;
    }
  }

  if (opt->getValue('m') != NULL) {
    rows = atoi(opt->getValue('m'));
  }

  if (opt->getValue('n') != NULL) {
    facs = atoi(opt->getValue('n'));
  }

  if (opt->getValue('p') != NULL) {
    fac_rows = opt->getValue('p');
  }
  
  if (opt->getValue('q') != NULL) {
    fac_cols = opt->getValue('q');
  }
  
  if (opt->getValue('t') != NULL) {
    type = opt->getValue('t');
  }

  checkResults = opt->getFlag('c');

  if (opt->getValue('r') != NULL) {
    runs = atoi(opt->getValue('r'));
  }

  if (opt->getValue('w') != NULL) {
    warmup = atoi(opt->getValue('w'));
  }

  if (opt->getValue("opx") != NULL) {
    char* str = opt->getValue("opx");
    if (strcmp(str, "N") == 0) {
      opx = fastKronOp_N;
    } else if (strcmp(str, "T") == 0) {
      opx = fastKronOp_T;
    }    
  }

  if (opt->getValue("opf") != NULL) {
    char* str = opt->getValue("opf");
    if (strcmp(str, "N") == 0) {
      opf = fastKronOp_N;
    } else if (strcmp(str, "T") == 0) {
      opf = fastKronOp_T;
    }
  }

  tune = opt->getFlag("tune");
  useUVA = opt->getFlag('u');
  if (useUVA == true) {
    printf("UVA is not supported\n");
    return 0;
  }

  if (opt->getValue("gpuLocalKrons") != NULL)
    gpuLocalKrons = atoi(opt->getValue("gpuLocalKrons"));
  if (opt->getValue("gpus") != NULL)
    gpus = atoi(opt->getValue("gpus"));
  if (opt->getValue("GM") != NULL)
    gpuInRows = atoi(opt->getValue("GM"));
  if (opt->getValue("GK") != NULL)
    gpuInCols = atoi(opt->getValue("GK"));
  if (gpus > 1) multiGPU = true;

  useFusion = opt->getFlag("fuse");

  if (useUVA) {
    if (gpuInRows <= 0 || gpuLocalKrons <= 0) {
      printf("Invalid gpurows %d , gpuLocalKrons %d\n", gpuInRows, gpuLocalKrons);
      return 1;
    }
  }

  if (rows <= 0 || type == NULL || runs <= 0 || facs <= 0 || fac_rows == NULL || fac_cols == NULL) {
    printf("Invalid value rows: %d, facs %d, fac_rows %s, fac_cols %s, type %p, runs %d\n", rows, facs, fac_rows, fac_cols, type, runs);
    return 1;
  }

  if (multiGPU) {
    if (gpus < 1 || ((gpuInRows != 0 and gpuInCols != 0) && gpuInRows * gpuInCols != gpus)) {
      printf("GM * GK != gpus (%d != %d)\n", gpuInRows * gpuInCols, gpus);
      return 1;
    }
    if (gpuLocalKrons > facs) {
      printf("Invalid Local Krons (%d) > Total Facs (%d)\n", gpuLocalKrons, facs);
    }
  }

  uint KP_MAT_N[facs];
  uint KP_MAT_K[facs];
  uint N = 1;
  uint K = 1;
  if (parseStringToIntegers(fac_cols, KP_MAT_N, facs) == false) {
    printf("Less than expected '%d' columns in '%s'\n", facs, fac_cols);
    return 1;
  }
  if (parseStringToIntegers(fac_rows, KP_MAT_K, facs) == false) {
    printf("Less than expected '%d' columns in '%s'\n", facs, fac_rows);
    return 1;
  }
  for (uint i = 0; i < (uint)facs; i++) {
    N *= KP_MAT_N[i];
    K *= KP_MAT_K[i];
  }
  
  printf("Doing KronMatmul of X[%d, %d] with ", rows, K);

  for (uint i = 0; i < (uint)facs; i++) {
    printf("F_%d [%d, %d] x ", i, KP_MAT_K[i], KP_MAT_N[i]);
  }
  printf("to produce Y[%d, %d]\n", rows, N);

  bool status = false;
  if (strcmp(type, "float") == 0)
    status = run<float>(rows, N, K, facs, KP_MAT_N, KP_MAT_K, opx, opf, runs, warmup, useUVA, gpuInRows, gpuInCols, gpus, gpuLocalKrons, checkResults, useFusion, tune, backend, false);
  // else if (strcmp(type, "int") == 0)
  //   status = run<int>(rows, N, K, facs, KP_MAT_N, KP_MAT_K, runs, warmup, useUVA, 
  //                     gpuInRows, gpuInCols, gpus, gpuLocalKrons, checkResults, useFusion, tune, false);
  // else if (strcmp(type, "double") == 0)
  //   status = run<double>(rows, N, K, facs, KP_MAT_N, KP_MAT_K, runs, warmup, useUVA, gpuInRows, gpuInCols, gpus, gpuLocalKrons, checkResults, useFusion, tune, false);
  else
    printf("type not supported %s\n", type);

  if (!status) return 1;

  return 0;
}