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
    for colsA in pow_range(MinColsA, MaxColsA):
        CRegCols = 1
        if colsA//NumThreads <= 1:
            CRegCols = 1
        else:
            CRegCols = min(16, min(colsA//NumThreads, kronRows))
        
        if CRegCols > 8:
            CRegRows = 1
        else:
            CRegRows = min(max((colsA//NumThreads)//CRegCols, 1), 8//CRegCols)

        Configs[kronRows][colsA] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": CRegRows, "CRegCols": CRegCols, "SharedTileKronRows": 32, "MaxTileKronCols": kronRows}


Configs[2][64] = {"NumThreads": 64, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 2}
Configs[2][128] = {"NumThreads": 128, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 2}
Configs[2][256] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 2}
Configs[2][1024] = {"NumThreads": 512, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 2, "SharedTileKronRows": 32, "MaxTileKronCols": 2}
Configs[2][2048] = {"NumThreads": 1024, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 2, "SharedTileKronRows": 32, "MaxTileKronCols": 2}

# KernelInfo{(void*)kronGemmKernel<T, VecT, 512, RowParallelismTy::Low, 1, RowModTileIsZero, 1024, 2, 2, 2, K_EQUALS_VAR, 1, 1, 2, 32>,512, 2, 2, 2, 1024, 1, 2},\
# KernelInfo{(void*)kronGemmKernel<T, VecT, 1024, RowParallelismTy::Low, 1, RowModTileIsZero, 2048, 2, 2, 2, K_EQUALS_VAR, 1, 1, 2, 32>,1024, 2, 2, 2, 2048, 1, 2},\

Configs[4][64] = {"NumThreads": 64, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 4}
Configs[4][4096] = {"NumThreads": 1024, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 4, "SharedTileKronRows": 32, "MaxTileKronCols": 4}

Configs[8][64] = {"NumThreads": 64, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 8}
Configs[8][512] = {"NumThreads": 64, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 1, "SharedTileKronRows": 32, "MaxTileKronCols": 8}
Configs[8][4096] = {"NumThreads": 512, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 8, "SharedTileKronRows": 32, "MaxTileKronCols": 8}

#KernelInfo{(void*)kronGemmKernel<T, VecT, 512, RowParallelismTy::Low, 1, RowModTileIsZero, 4096, 8, 8, 8, K_EQUALS_VAR, 1, 1, 8, 32>,512, 8, 8, 8, 4096, 1, 8},\
# KernelInfo{(void*)kronGemmKernel<T, VecT, 1024, RowParallelismTy::Low, 1, RowModTileIsZero, 4096, 4, 4, 4, K_EQUALS_VAR, 1, 1, 4, 32>,1024, 4, 4, 4, 4096, 1, 4},\

for kronRows in pow_range(32, MaxKronRows):
    Configs[kronRows] = {}

for colsA in pow_range(MinColsA, MaxColsA):
    Configs[32][colsA] = {"NumThreads": 256, "RowsTileA": 2, "CRegRows": 1, "CRegCols": 4, "SharedTileKronRows": 32, "MaxTileKronCols": 32}
    if colsA < 4096:
        Configs[64][colsA] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 8, "SharedTileKronRows": 32, "MaxTileKronCols": 64}
    elif colsA == 4096:
        Configs[64][colsA] = {"NumThreads": 128, "RowsTileA": 1, "CRegRows": 2, "CRegCols": 16, "SharedTileKronRows": 32, "MaxTileKronCols": 64}
    elif colsA == 8192:
        Configs[64][colsA] = {"NumThreads": 128, "RowsTileA": 1, "CRegRows": 4, "CRegCols": 16, "SharedTileKronRows": 32, "MaxTileKronCols": 64}
    else:
        Configs[64][colsA] = {"NumThreads": 128, "RowsTileA": 1, "CRegRows": 2, "CRegCols": 16, "SharedTileKronRows": 32, "MaxTileKronCols": 64}
    Configs[128][colsA] = {"NumThreads": 128, "RowsTileA": 1, "CRegRows": 4, "CRegCols": 16, "SharedTileKronRows": 32, "MaxTileKronCols": 128}
    Configs[256][colsA] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 32, "SharedTileKronRows": 32, "MaxTileKronCols": 64}
    Configs[512][colsA] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 32, "SharedTileKronRows": 32, "MaxTileKronCols": 128}
    Configs[1024][colsA] = {"NumThreads": 256, "RowsTileA": 1, "CRegRows": 1, "CRegCols": 32, "SharedTileKronRows": 32, "MaxTileKronCols": 256}

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
    
    f.write("static uint MaxTileRowsA[] = {")
    for d in pow_range(MinKronRows, MaxKronRows):
        config = Configs[d]
        config = config[MaxColsA]
        rowsTileA = config["RowsTileA"]
        f.write(f"{rowsTileA}")
        if d != MaxKronRows:
            f.write(", ")
    f.write("};\n\n")
    f.write("static uint MaxTileKronCols[] = {")
    for d in pow_range(MinKronRows, MaxKronRows):
        config = Configs[d][MaxColsA]
        tileKronCols = config["MaxTileKronCols"]
        f.write(f"{tileKronCols}")
        if d != MaxKronRows:
            f.write(", ")
    f.write("};\n\n")
    
    f.write("#define KERNEL_DECL(T, VecT, RowModTileIsZero, K_EQUALS_VAR) \\\n")
    contents = ""
    for colsA in AllColsA:
        for kronRows in AllKronRows:
            config = Configs[kronRows][colsA]
            if colsA < kronRows or not isValid(colsA, kronRows, config) or tooMuchSharedMem(colsA, kronRows):
                contents += "KernelInfo{NULL}"
            else:
                rowsTileA = config["RowsTileA"]
                regRows = config["CRegRows"]
                regCols = config["CRegCols"]
                tileKronCols = config["MaxTileKronCols"]
                sharedTileKronRows = config["SharedTileKronRows"]
                numThreads = config["NumThreads"]
                contents += "KernelInfo{"+ \
                    f"(void*)kronGemmKernel<T, VecT, {numThreads}, RowParallelismTy::Low, {rowsTileA}, RowModTileIsZero, {colsA}, {kronRows}, {kronRows}, {tileKronCols}, K_EQUALS_VAR, 1, {regRows}, {regCols}, {sharedTileKronRows}>,"+\
                    f"{numThreads}, {kronRows}, {kronRows}, {tileKronCols}, {colsA}, {regRows}, {regCols}"+ "}"
            contents += ",\\\n"

    #Remove last comma and backslash
    contents = contents[:contents.rfind(",")]
    contents += "\n"
    f.write(contents)