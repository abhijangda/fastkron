import os
import math

def pow_range(start, end):
    l = []
    while start <= end:
        l += [start]
        start = start * 2
    return l

MinKronRows = 2
MaxKronRows = 256

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

        Configs[kronRows][colsA] = {"RowsTileA": 1, "CRegRows": CRegRows, "CRegCols": CRegCols, "SharedTileKronRows": 32, "MaxTileKronCols": 32}

Configs[32] = {}
Configs[64] = {}
Configs[128] = {}
Configs[256] = {}

for colsA in pow_range(MinColsA, MaxColsA):
    Configs[32][colsA] = {"RowsTileA": 2, "CRegRows": 1, "CRegCols": 4, "SharedTileKronRows": 32, "MaxTileKronCols": 32}
    Configs[64][colsA] = {"RowsTileA": 2, "CRegRows": 1, "CRegCols": 8, "SharedTileKronRows": 16, "MaxTileKronCols": 64}
    Configs[128][colsA] = {"RowsTileA": 2, "CRegRows": 1, "CRegCols": 32, "SharedTileKronRows": 16, "MaxTileKronCols": 64}
    Configs[256][colsA] = {"RowsTileA": 1, "CRegRows": 1, "CRegCols": 32, "SharedTileKronRows": 32, "MaxTileKronCols": 64}

def tooMuchSharedMem(colsA, kronRows):
    return (colsA == 65536 and kronRows <= 128) or \
           (colsA == 32768 and kronRows <= 128) or \
           (colsA == 16384 and kronRows <= 4)

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
    
    f.write("#define KERNEL_DECL(T, VecT, K_EQUALS_VAR) \\\n")
    contents = ""
    for colsA in AllColsA:
        for kronRows in AllKronRows:
            if colsA < kronRows or not math.log(colsA, kronRows).is_integer() or tooMuchSharedMem(colsA, kronRows):
                contents += "    NULL"
            else:
                config = Configs[kronRows][colsA]
                rowsTileA = config["RowsTileA"]
                regRows = config["CRegRows"]
                regCols = config["CRegCols"]
                tileKronCols = config["MaxTileKronCols"]
                sharedTileKronRows = config["SharedTileKronRows"]
                contents += f"    (void*)kronGemmKernel<T, VecT, N_THREADS, 1, {rowsTileA}, {colsA}, {kronRows}, {kronRows}, {tileKronCols}, K_EQUALS_VAR, 1, {regRows}, {regCols}, {sharedTileKronRows}>"
        
            contents += ",\\\n"

    #Remove last comma and backslash
    contents = contents[:contents.rfind(",")]
    contents += "\n"
    f.write(contents)