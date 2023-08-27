#include "anyoption.h"
#include "testBase.h"

/**************************************************
              Main Function
***************************************************/
int main(int argc, char* argv[]) {  
  int batch = 0;
  int facs = 0;
  int size = 0;
  char* type = NULL;
  bool checkResults = false;
  int runs = 0;
  int warmup = 0;
  bool useUVA = false;
  int gpurows = 0;
  int maxkronbatch = 0;
  int nummaxkronbatch = 0;
  int gpus = 1;
  bool useFusion = true;

  AnyOption *opt = new AnyOption();

  opt->addUsage("usage: ");
  opt->addUsage("batch: Size of Batch");
  opt->addUsage("facs:  Number of Kron Factors");
  opt->addUsage("size:  Row and cols of each Kron Factor");
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
  opt->setOption("size", 's');
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

  if (opt->getValue('s') != NULL) {
    size = atoi(opt->getValue('s'));
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

  useUVA = opt->getFlag('u');
  if (opt->getValue("gpurows") != NULL)
    gpurows = atoi(opt->getValue("gpurows"));
  if (opt->getValue("maxkronbatch") != NULL)
    maxkronbatch = atoi(opt->getValue("maxkronbatch"));
  if (opt->getValue("nummaxkronbatch") != NULL)
    nummaxkronbatch = atoi(opt->getValue("nummaxkronbatch"));
  if (opt->getValue("gpus") != NULL)
    gpus = atoi(opt->getValue("gpus"));
  useFusion = opt->getFlag("fuse");

  if (useUVA) {
    if (gpurows <= 0 || maxkronbatch <= 0 || nummaxkronbatch <= 0) {
      printf("Invalid gpurows %d , maxkronbatch %d nummaxkronbatch %d\n", gpurows, maxkronbatch, nummaxkronbatch);
      return 1;
    }
  }

  if (batch <= 0 || facs <= 0 || size <= 0 || type == NULL || runs <= 0) {
    printf("Invalid value batch: %d, facs %d, size %d, type %p, runs %d\n", batch, facs, size, type, runs);
    return 1;
  }

  uint KP_MAT_N[facs];
  uint KP_MAT_K[facs];
  uint N = 1;
  uint K = 1;
  for (uint i = 0; i < (uint)facs; i++) {
    N *= size;
    K *= size;
    KP_MAT_K[i] = KP_MAT_N[i] = size;
  }
  
  bool status = false;
  if (strcmp(type, "float") == 0)
    status = run<float>(batch, N, K, facs, KP_MAT_N, KP_MAT_K, runs, warmup, useUVA, gpurows, maxkronbatch, nummaxkronbatch, gpus, checkResults, useFusion, false);
  else if (strcmp(type, "int") == 0)
    status = run<int>(batch, N, K, facs, KP_MAT_N, KP_MAT_K, runs, warmup, useUVA, gpurows, maxkronbatch, nummaxkronbatch, gpus, checkResults, useFusion, false);
  else if (strcmp(type, "double") == 0)
    status = run<double>(batch, N, K, facs, KP_MAT_N, KP_MAT_K, runs, warmup, useUVA, gpurows, maxkronbatch, nummaxkronbatch, gpus, checkResults, useFusion, false);
  else
    printf("type not supported %s\n", type);

  if (!status) return 1;

  return 0;
}