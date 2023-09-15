#include "anyoption.h"
#include "testBase.h"

#include <sstream>

bool parseStringToIntegers(char* str, uint array[], const int numInts) {
  int parsedInts = 0;
  
  std::string stdstr(str);
  std::cout << stdstr << std::endl;
  std::stringstream stream(stdstr);
  uint n;
  char comma;
  while(stream >> n){
    array[parsedInts] = n;
    stream >> comma;
    if (comma != ',')
      break;
    parsedInts++;
  }

  if (parsedInts != numInts) return false;

  return true;
}

/**************************************************
              Main Function
***************************************************/
int main(int argc, char* argv[]) {  
  int batch = 0;
  int facs = 0;
  char* fac_rows = NULL;
  char* fac_cols = NULL;
  char* type = NULL;
  bool checkResults = false;
  int runs = 0;
  int warmup = 0;
  bool useUVA = false;
  int gpurows = 0;
  int maxkronbatch = 0;
  int nummaxkronbatch = 0;
  int gpus = 1;
  bool multiGPU = false;
  bool useFusion = true;
  bool tune = false;
  AnyOption *opt = new AnyOption();

  opt->addUsage("usage: ");
  opt->addUsage("batch: Size of Batch");
  opt->addUsage("facs:  Number of Kron Factors");
  opt->addUsage("fac_rows: Rows of each Kron Factor separated by space");
  opt->addUsage("fac_cols: Cols of each Kron Factor separated by space");
  opt->addUsage("type:  Type of matrices (float, int, half, double)");
  opt->addUsage("check: Check results for first run");
  opt->addUsage("runs:  Number of runs");
  opt->addUsage("warmup:  Number of warmup runs");
  opt->addUsage("uva: Allocate and run using NVIDIA UVA");
  opt->addUsage("gpurows: Rows for temp on GPU. valid only with uva");
  opt->addUsage("maxkronbatch: Factors batch per inner iteration. valid only with uva");
  opt->addUsage("nummaxkronbatch");

  opt->setOption("batch", 'b');
  opt->setOption("facs", 'f');
  opt->setOption("fac_rows", 'p');
  opt->setOption("fac_cols", 'q');
  opt->setOption("type", 't');
  opt->setOption("runs", 'r');
  opt->setOption("warmup", 'w');
  
  opt->setFlag("check", 'c');
  opt->setFlag("uva", 'u');
  opt->setOption("gpurows");
  opt->setOption("maxkronbatch");
  opt->setOption("nummaxkronbatch");
  opt->setOption("gpus");

  opt->setFlag("fuse");
  opt->setFlag("tune");

  opt->processCommandArgs(argc, argv);
  
  if (!opt->hasOptions()) { /* print usage if no options */
    opt->printUsage();
    delete opt;
    return 1;
  }

  if (opt->getValue('b') != NULL) {
    batch = atoi(opt->getValue('b'));
  }

  if (opt->getValue('f') != NULL) {
    facs = atoi(opt->getValue('f'));
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

  tune = opt->getFlag("tune");
  useUVA = opt->getFlag('u');
  if (useUVA == true) {
    printf("UVA is not supported\n");
    return 0;
  }
  if (opt->getValue("gpurows") != NULL)
    gpurows = atoi(opt->getValue("gpurows"));
  if (opt->getValue("maxkronbatch") != NULL)
    maxkronbatch = atoi(opt->getValue("maxkronbatch"));
  if (opt->getValue("nummaxkronbatch") != NULL)
    nummaxkronbatch = atoi(opt->getValue("nummaxkronbatch"));
  if (opt->getValue("gpus") != NULL)
    gpus = atoi(opt->getValue("gpus"));
  if (gpus > 1) multiGPU = true;

  useFusion = opt->getFlag("fuse");

  if (useUVA) {
    if (gpurows <= 0 || maxkronbatch <= 0 || nummaxkronbatch <= 0) {
      printf("Invalid gpurows %d , maxkronbatch %d nummaxkronbatch %d\n", gpurows, maxkronbatch, nummaxkronbatch);
      return 1;
    }
  }

  if (batch <= 0 || facs <= 0 || fac_rows == NULL || fac_cols == NULL || type == NULL || runs <= 0) {
    printf("Invalid value batch: %d, facs %d, fac_rows %s, fac_cols %s, type %p, runs %d\n", batch, facs, fac_rows, fac_cols, type, runs);
  if (multiGPU) {
    // if (gpurows <= 0 || maxkronbatch <= 0 || nummaxkronbatch <= 0) {
    //   printf("Invalid gpurows %d , maxkronbatch %d nummaxkronbatch %d\n", gpurows, maxkronbatch, nummaxkronbatch);
    //   return 1;
    // }
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
  
  bool status = false;
  if (strcmp(type, "float") == 0)
    status = run<float>(batch, N, K, facs, KP_MAT_N, KP_MAT_K, runs, warmup, useUVA, gpurows, maxkronbatch, nummaxkronbatch, gpus, checkResults, useFusion, tune, false);
  else if (strcmp(type, "int") == 0)
    status = run<int>(batch, N, K, facs, KP_MAT_N, KP_MAT_K, runs, warmup, useUVA, gpurows, maxkronbatch, nummaxkronbatch, gpus, checkResults, useFusion, tune, false);
  else if (strcmp(type, "double") == 0)
    status = run<double>(batch, N, K, facs, KP_MAT_N, KP_MAT_K, runs, warmup, useUVA, gpurows, maxkronbatch, nummaxkronbatch, gpus, checkResults, useFusion, tune, false);
  else
    printf("type not supported %s\n", type);

  if (!status) return 1;

  return 0;
}