import os

def pow_range(start, end):
    l = []
    while start <= end:
        l += [start]
        start = start * 2
    return l

MinColsA = 16
MaxColsA = 4096

MinKronRows = 2
MaxKronRows = 64

MaxKronRowsTile = 32

AllColsA    = pow_range(MinColsA, MaxColsA)
AllKronRows = pow_range(MinKronRows, MaxKronRows)
RowTilesA   = {}
for kronRows in pow_range(MinKronRows, 16):
    RowTilesA[kronRows] = 1
RowTilesA[32] = 2
RowTilesA[64] = 2
RowTilesA[128] = 2
RowTilesA[256] = 1

with open("kernel_decl.inc", "w") as f:
    f.write("static uint MaxTileRowsA[] = {")
    for d in pow_range(MinKronRows, MaxKronRows):
        f.write(f"{RowTilesA[d]}")
        if d != MaxKronRows:
            f.write(", ")
    f.write("};\n\n")
    f.write("#define KERNEL_DECL(T, VecT, K_EQUALS_VAR) \\\n")
    contents = ""
    for colsA in AllColsA:
        for kronRows in AllKronRows:
            if colsA < kronRows:
                contents += "    NULL"
            else:
                contents += f"    (void*)kronGemmKernel<T, VecT, N_THREADS,1,{RowTilesA[kronRows]},{colsA},{kronRows},{kronRows},{MaxKronRowsTile},K_EQUALS_VAR,1>"
        
            contents += ",\\\n"

    #Remove last comma and backslash
    contents = contents[:contents.rfind(",")]
    contents += "\n"
    f.write(contents)