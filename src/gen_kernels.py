import os
import math

def pow_range(start, end):
    l = []
    while start <= end:
        l += [start]
        start = start * 2
    return l

MinKronRows = 2
MaxKronRows = 1024

MinColsA = 16
MaxColsA = 65536

NumThreads = 256

AllColsA    = pow_range(MinColsA, MaxColsA)
AllKronRows = pow_range(MinKronRows, MaxKronRows)
Configs   = {}

for kronRows in pow_range(MinKronRows, 16):
    Configs[kronRows] = {}
    # for colsA in pow_range(MinColsA, 8192):
        # CRegCols = 1
        # if colsA//NumThreads <= 1:
        #     CRegCols = 1
        # else:
        #     CRegCols = min(16, min(colsA//NumThreads, kronRows))
        
        # if CRegCols > 8:
        #     CRegRows = 1
        # else:
        #     CRegRows = min(max((colsA//NumThreads)//CRegCols, 1), 8//CRegCols)

        # Configs[kronRows][colsA] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": CRegRows, "CRegCols": CRegCols, "SharedTileKronRows": 32, "MaxTileKronCols": kronRows, "NumFusedKernels": 1}


Configs[2][64] = {"NumThreads": 64, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 2, "NumFusedKernels": 1}
Configs[2][128] = {"NumThreads": 128, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 2, "NumFusedKernels": 1}
Configs[2][256] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 2, "NumFusedKernels": 1}
Configs[2][1024] = {"NumThreads": 512, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 2, "SharedTileKronRows": 32, "MaxTileKronCols": 2, "NumFusedKernels": 1}
Configs[2][2048] = {"NumThreads": 1024, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 2, "SharedTileKronRows": 32, "MaxTileKronCols": 2, "NumFusedKernels": 1}

Configs[4][64] = {"NumThreads": 64, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 4, "NumFusedKernels": 1}
Configs[4][256] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 4, "NumFusedKernels": 1}
Configs[4][1024] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 4, "SharedTileKronRows": 32, "MaxTileKronCols": 4, "NumFusedKernels": 1}

Configs[8][64] = {"NumThreads": 64, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 8, "NumFusedKernels": 1}
Configs[8][512] = {"NumThreads": 512, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 8, "NumFusedKernels": 1}
Configs[8][4096] = {"NumThreads": 512, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 8, "SharedTileKronRows": 32, "MaxTileKronCols": 8, "NumFusedKernels": 1}

Configs[16][256] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 16, "NumFusedKernels": 1}
Configs[16][4096] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 16, "SharedTileKronRows": 32, "MaxTileKronCols": 16, "NumFusedKernels": 1}

for kronRows in pow_range(32, MaxKronRows):
    Configs[kronRows] = {}

Configs[32][1024] = {"NumThreads": 256, "RowsTileA": 2, "CRegRows": 1, "CRegCols": 4, "SharedTileKronRows": 32, "MaxTileKronCols": 32, "NumFusedKernels": 1}
Configs[64][64] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 8, "SharedTileKronRows": 32, "MaxTileKronCols": 64, "NumFusedKernels": 1}
Configs[64][4096] = {"NumThreads": 128, "RowsTileA": 1, "CRegRows": 2, "CRegCols": 16, "SharedTileKronRows": 32, "MaxTileKronCols": 64, "NumFusedKernels": 1}
Configs[64][8192] = {"NumThreads": 128, "RowsTileA": 1, "CRegRows": 4, "CRegCols": 16, "SharedTileKronRows": 32, "MaxTileKronCols": 64, "NumFusedKernels": 1}

Configs[128][8192] = {"NumThreads": 128, "RowsTileA": 1, "CRegRows": 4, "CRegCols": 16, "SharedTileKronRows": 32, "MaxTileKronCols": 128, "NumFusedKernels": 1}
Configs[256][32768] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 32, "SharedTileKronRows": 32, "MaxTileKronCols": 64, "NumFusedKernels": 1}
Configs[512][32768] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 32, "SharedTileKronRows": 32, "MaxTileKronCols": 128, "NumFusedKernels": 1}
Configs[1024][65536] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 32, "SharedTileKronRows": 32, "MaxTileKronCols": 256, "NumFusedKernels": 1}
    
def tooMuchSharedMem(colsA, kronRows):
    return (colsA == 65536 and kronRows <= 128) or \
           (colsA == 32768 and kronRows <= 128) or \
           (colsA > 4096 and kronRows <= 4) #for KronMat = 4, 4096 works best even for larger sizes

def isValid(colsA, kronRows, config):
    if (colsA == 65536 and kronRows == 1024):
        return True
    if kronRows >= 256 and kronRows < 1024:
        if colsA == 32768:
            return True
        return False
    if kronRows == 64:
        if colsA == 8192:
            return True
    if kronRows == 128:
        if colsA == 8192:
            return True
        else:
            return False
    if kronRows == 2 and colsA >= 4096:
        return False
    return math.log(colsA, kronRows).is_integer()

with open("kernel_decl.inc", "w") as f:
    f.writelines([f"#define MAX_K {MaxColsA}\n"])
    f.writelines([f"#define MIN_K {MinColsA}\n"])
    
    f.writelines([f"#define MIN_KP_K {MinKronRows}\n"])
    f.writelines([f"#define MAX_KP_K {MaxKronRows}\n"])
    
    # f.write("static uint MaxTileRowsA[] = {")
    # for d in pow_range(MinKronRows, MaxKronRows):
    #     config = Configs[d]
    #     if MaxColsA in config:
    #         config = config[MaxColsA]
    #         rowsTileA = config["RowsTileA"]
    #         f.write(f"{rowsTileA}")
    #     else:
    #         f.write("1")
    #     if d != MaxKronRows:
    #         f.write(", ")
    # f.write("};\n\n")
    # f.write("static uint MaxTileKronCols[] = {")
    # for d in pow_range(MinKronRows, MaxKronRows):
    #     config = Configs[d][MaxColsA]
    #     tileKronCols = config["MaxTileKronCols"]
    #     f.write(f"{tileKronCols}")
    #     if d != MaxKronRows:
    #         f.write(", ")
    # f.write("};\n\n")
    
    f.write("#define KERNEL_DECL(T, VecT, ElemType, RowModTileIsZero, K_EQUALS_VAR) \\\n")
    contents = ""
    for colsA in AllColsA:
        for kronRows in AllKronRows:
            try:
                config = Configs[kronRows][colsA]
                rowsTileA = config["RowsTileA"]
                regRows = config["CRegRows"]
                regCols = config["CRegCols"]
                tileKronCols = config["MaxTileKronCols"]
                sharedTileKronRows = config["SharedTileKronRows"]
                numThreads = config["NumThreads"]
                numFusedKerns = config["NumFusedKernels"]
                contents += "KernelInfo{"+ \
                    f"(void*)kronGemmKernel<T, VecT, {numThreads}, RowParallelismTy::Low, {rowsTileA}, RowModTileIsZero, {colsA}, {kronRows}, {kronRows}, {tileKronCols}, K_EQUALS_VAR, 1, {regRows}, {regCols}, {sharedTileKronRows}, {numFusedKerns}>,"+\
                    f"{numThreads}, {kronRows}, {kronRows}, {tileKronCols}, {rowsTileA}, {colsA}, {regRows}, {regCols}, {numFusedKerns}, ElemType, RowModTileIsZero, K_EQUALS_VAR"+ "}"
                contents += ",\\\n"
            except:
                pass
    #Remove last comma and backslash
    contents = contents[:contents.rfind(",")]
    contents += "\n"
    f.write(contents)