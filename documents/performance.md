Below tables show the throughput in Tera FLOPs achieved by FastKron without fusion, FastKron, GPyTorch (using Shuffle Algorithm), and speedup.
First column shows the shape of KMM/MKM in the format M_PxQ^N, such that, M is the number of rows of $X$ in MKM or columns of $X$ in KMM, PxQ is the size of each Kronecker Factor and N is the number of Kronecker Factors.
The last column of speedup shows that FastKron performs orders of magnitude faster than GPyTorch on different sizes, float and double data types, operation on X and Factors (N for no-transpose and T for transpose), when running on wide range of CPUs and GPUs.

## NVIDIA A100 80GB
<b> MKM FLOAT OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  3405.115 | 5134.676 | 762.856 | 6.731
1024_8x8^6  |  3555.577 | 6486.911 | 645.917 | 10.043
1024_16x16^4  |  6924.318 | 10604.531 | 1465.288 | 7.237
1024_16x16^5  |  7046.210 | 9832.221 | 1439.027 | 6.833
1024_32x32^3  |  12670.903 | 12603.077 | 3678.209 | 3.426
1024_32x32^4  |  13670.581 | 13669.891 | 2741.247 | 4.987
1024_64x64^2  |  13014.384 | 13014.384 | 3950.526 | 3.294
1024_64x64^3  |  16922.982 | 16922.982 | 4468.697 | 3.787
1024_128x128^2  |  16279.483 | 16279.483 | 9529.411 | 1.708
320_128x128^3  |  17282.226 | 17282.226 | 7715.359 | 2.240
16_8x8^8  |  3510.701 | 6275.776 | 886.345 | 7.081
16_16x16^6  |  7005.529 | 10728.421 | 2262.916 | 4.741
16_32x32^5  |  13554.672 | 13554.564 | 4903.825 | 2.764
16_64x64^4  |  16910.607 | 16910.607 | 8737.537 | 1.935
16_128x128^3  |  16773.939 | 16773.939 | 11224.157 | 1.494

--------------------
<b> MKM FLOAT  OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  3313.164 | 4559.606 | 930.650 | 4.899
1024_8x8^6  |  3467.139 | 5500.270 | 958.357 | 5.739
1024_16x16^4  |  6569.772 | 7888.592 | 2579.935 | 3.058
1024_16x16^5  |  6791.268 | 9182.571 | 2605.958 | 3.524
1024_32x32^3  |  11735.263 | 11702.857 | 5667.281 | 2.065
1024_32x32^4  |  12293.974 | 12292.690 | 5861.319 | 2.097
1024_64x64^2  |  11147.342 | 11147.342 | 4276.111 | 2.607
1024_64x64^3  |  14949.780 | 14949.780 | 10250.208 | 1.458
1024_128x128^2  |  15549.070 | 15549.070 | 12308.280 | 1.263
320_128x128^3  |  16655.353 | 16655.353 | 12910.262 | 1.290
16_8x8^8  |  3504.094 | 5665.434 | 925.190 | 6.124
16_16x16^6  |  6952.756 | 8945.950 | 2441.797 | 3.664
16_32x32^5  |  12756.977 | 12758.038 | 5429.106 | 2.350
16_64x64^4  |  15620.632 | 15620.632 | 9737.250 | 1.604
16_128x128^3  |  16720.584 | 16720.584 | 12285.057 | 1.361

-------------------------
<b> MKM DOUBLE OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  1756.079 | 3269.623 | 386.074 | 8.469
1024_8x8^6  |  1782.167 | 4353.150 | 381.243 | 11.418
1024_16x16^4  |  3490.957 | 6185.335 | 907.458 | 6.816
1024_16x16^5  |  3479.987 | 5442.488 | 965.255 | 5.638
1024_32x32^3  |  6531.380 | 6367.180 | 1959.706 | 3.249
1024_32x32^4  |  6759.057 | 6759.614 | 1976.190 | 3.421
1024_64x64^2  |  6447.639 | 6447.639 | 3587.382 | 1.797
1024_64x64^3  |  8069.379 | 8069.379 | 3406.136 | 2.369
1024_128x128^2  |  7728.160 | 7728.160 | 6987.742 | 1.106
320_128x128^3  |  8174.192 | 8174.192 | 7185.855 | 1.138
16_8x8^8  |  1755.486 | 3776.035 | 455.316 | 8.293
16_16x16^6  |  3423.881 | 6167.496 | 1273.140 | 4.844
16_32x32^5  |  6618.085 | 6618.474 | 2852.397 | 2.320
16_64x64^4  |  8046.702 | 8046.702 | 5376.189 | 1.497
16_128x128^3  |  7935.139 | 7935.139 | 8607.249 | 0.922

---------------------------------
<b>  MKM DOUBLE OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  1689.285 | 2477.389 | 457.661 | 5.413
1024_8x8^6  |  1729.916 | 3081.827 | 461.910 | 6.672
1024_16x16^4  |  3271.515 | 4110.624 | 1314.171 | 3.128
1024_16x16^5  |  3315.973 | 5048.093 | 1329.679 | 3.796
1024_32x32^3  |  5689.602 | 5582.184 | 3049.222 | 1.831
1024_32x32^4  |  6046.867 | 6034.902 | 3078.602 | 1.960
1024_64x64^2  |  5310.392 | 5310.392 | 3938.090 | 1.348
1024_64x64^3  |  6408.131 | 6408.131 | 5464.497 | 1.173
1024_128x128^2  |  7100.655 | 7100.655 | 8637.100 | 0.822
320_128x128^3  |  7553.576 | 7553.576 | 9002.660 | 0.839
16_8x8^8  |  1740.637 | 3097.981 | 455.464 | 6.802
16_16x16^6  |  3365.959 | 4764.067 | 1301.271 | 3.661
16_32x32^5  |  6277.124 | 6278.805 | 2988.649 | 2.101
16_64x64^4  |  7027.916 | 7027.916 | 5297.447 | 1.327
16_128x128^3  |  7503.304 | 7503.304 | 9085.944 | 0.826

------------------------
<b> KMM FLOAT OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  3190.046 | 5656.792 | 934.202 | 6.055
1024_8x8^6  |  3331.489 | 7071.322 | 958.954 | 7.374
1024_16x16^4  |  5852.634 | 9587.756 | 2570.915 | 3.729
1024_16x16^5  |  5943.061 | 8973.883 | 2608.714 | 3.440
1024_32x32^3  |  10780.044 | 10824.097 | 5726.371 | 1.890
1024_32x32^4  |  11372.699 | 11371.503 | 5875.592 | 1.935
1024_64x64^2  |  12374.333 | 12374.333 | 5315.864 | 2.328
1024_64x64^3  |  17599.370 | 17599.370 | 10256.871 | 1.716
1024_128x128^2  |  16666.091 | 16666.091 | 12456.367 | 1.338
320_128x128^3  |  17890.331 | 17890.331 | 12886.635 | 1.388
16_8x8^8  |  3428.374 | 7095.152 | 925.645 | 7.665
16_16x16^6  |  6622.039 | 11005.635 | 2485.442 | 4.428
16_32x32^5  |  12109.483 | 12101.844 | 5468.487 | 2.213
16_64x64^4  |  15579.092 | 15579.092 | 9708.616 | 1.605
16_128x128^3  |  16790.684 | 16790.684 | 12560.590 | 1.337

--------------------------
<b>  KMM FLOAT OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  3145.605 | 5895.077 | 765.595 | 7.700
1024_8x8^6  |  3275.588 | 5696.381 | 645.841 | 8.820
1024_16x16^4  |  5419.643 | 7643.025 | 1462.552 | 5.226
1024_16x16^5  |  5348.277 | 7639.039 | 1439.635 | 5.306
1024_32x32^3  |  9000.413 | 8992.774 | 3707.277 | 2.426
1024_32x32^4  |  9689.528 | 9689.771 | 2744.523 | 3.531
1024_64x64^2  |  11314.433 | 11314.433 | 4857.204 | 2.329
1024_64x64^3  |  15576.666 | 15576.666 | 4466.592 | 3.487
1024_128x128^2  |  14170.244 | 14170.244 | 9627.190 | 1.472
320_128x128^3  |  15694.999 | 15694.999 | 7702.307 | 2.038
16_8x8^8  |  3317.309 | 5670.619 | 886.847 | 6.394
16_16x16^6  |  6201.717 | 8288.665 | 2299.534 | 3.604
16_32x32^5  |  10057.535 | 10064.669 | 4945.179 | 2.035
16_64x64^4  |  14100.253 | 14100.253 | 8715.160 | 1.618
16_128x128^3  |  13588.626 | 13588.626 | 11476.332 | 1.184

-------------------------
<b> KMM DOUBLE  OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  1659.857 | 2582.141 | 458.847 | 5.627
1024_8x8^6  |  1698.026 | 3164.548 | 461.860 | 6.852
1024_16x16^4  |  3124.539 | 5487.716 | 1324.286 | 4.144
1024_16x16^5  |  3153.770 | 4775.232 | 1330.197 | 3.590
1024_32x32^3  |  5909.083 | 5708.506 | 3054.461 | 1.869
1024_32x32^4  |  6110.486 | 6115.805 | 3084.024 | 1.983
1024_64x64^2  |  6253.065 | 6253.065 | 4243.875 | 1.473
1024_64x64^3  |  7767.468 | 7767.468 | 5482.083 | 1.417
1024_128x128^2  |  7397.554 | 7397.554 | 8698.826 | 0.850
320_128x128^3  |  7714.455 | 7714.455 | 8998.006 | 0.857
16_8x8^8  |  1756.825 | 3294.412 | 455.429 | 7.234
16_16x16^6  |  3350.171 | 5887.398 | 1303.944 | 4.515
16_32x32^5  |  6398.748 | 6397.609 | 2992.629 | 2.138
16_64x64^4  |  7687.858 | 7687.858 | 5335.574 | 1.441
16_128x128^3  |  7682.183 | 7682.183 | 9110.144 | 0.843

-----------------------
<b> KMM DOUBLE OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  1596.540 | 2489.324 | 387.418 | 6.425
1024_8x8^6  |  1638.306 | 2834.427 | 381.253 | 7.435
1024_16x16^4  |  3050.478 | 5029.030 | 916.400 | 5.488
1024_16x16^5  |  3096.426 | 4620.647 | 966.430 | 4.781
1024_32x32^3  |  5188.534 | 5161.572 | 1965.607 | 2.626
1024_32x32^4  |  5486.753 | 5487.555 | 1976.489 | 2.776
1024_64x64^2  |  5564.866 | 5564.866 | 3812.732 | 1.460
1024_64x64^3  |  6773.883 | 6773.883 | 3420.721 | 1.980
1024_128x128^2  |  6345.698 | 6345.698 | 7032.754 | 0.902
320_128x128^3  |  6769.440 | 6769.440 | 7179.516 | 0.943
16_8x8^8  |  1706.086 | 2991.303 | 455.255 | 6.571
16_16x16^6  |  3279.674 | 5414.469 | 1275.622 | 4.245
16_32x32^5  |  5789.959 | 5789.959 | 2857.596 | 2.026
16_64x64^4  |  7033.162 | 7033.162 | 5429.898 | 1.295
16_128x128^3  |  6717.794 | 6717.794 | 8621.392 | 0.779

## NVIDIA V100 16GB

<b> MKM Float OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^6  |  1682.876 | 3759.602 | 375.948 | 10.000
256_16x16^5  |  3337.288 | 5328.665 | 757.737 | 7.032
256_32x32^4  |  6593.275 | 8556.934 | 1372.127 | 6.236
256_64x64^3  |  10434.721 | 10434.721 | 2346.806 | 4.446
128_128x128^3  |  11235.911 | 11235.911 | 4601.423 | 2.442

<b> MKM Float  OpX=T OpF=T</b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^6  |  1664.946 | 2937.221 | 622.227 | 4.720
256_16x16^5  |  3077.454 | 4719.378 | 1510.958 | 3.123
256_32x32^4  |  6266.449 | 6987.906 | 3145.493 | 2.222
256_64x64^3  |  9674.875 | 9674.875 | 6132.700 | 1.578
128_128x128^3  |  10518.161 | 10518.161 | 8795.341 | 1.196

<b> MKM Double OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^6  |  846.149 | 2324.644 | 196.513 | 11.829
256_16x16^5  |  1660.553 | 2386.194 | 364.606 | 6.545
256_32x32^4  |  2663.970 | 4375.653 | 913.897 | 4.788
256_64x64^3  |  5278.214 | 5278.214 | 1603.884 | 3.291
128_128x128^3  |  5441.296 | 5441.296 | 2874.262 | 1.893

<b> MKM Double OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^6  |  836.180 | 1580.696 | 253.192 | 6.243
256_16x16^5  |  1598.373 | 2258.864 | 663.694 | 3.403
256_32x32^4  |  3075.618 | 3191.944 | 1534.317 | 2.080
256_64x64^3  |  4385.731 | 4385.731 | 2993.453 | 1.465
128_128x128^3  |  5119.662 | 5119.662 | 4406.451 | 1.162

<b> KMM Float OpX=N OpX=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^6  |  1580.502 | 3968.443 | 623.175 | 6.368
256_16x16^5  |  3160.147 | 4883.716 | 1496.531 | 3.263
256_32x32^4  |  5704.117 | 5706.428 | 3148.400 | 1.812
256_64x64^3  |  9658.253 | 9658.253 | 6135.137 | 1.574
128_128x128^3  |  12068.812 | 12068.812 | 8785.851 | 1.374

<b> KMM Float OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^6  |  1403.923 | 3056.745 | 376.250 | 8.124
256_16x16^5  |  2712.493 | 3987.880 | 754.110 | 5.288
256_32x32^4  |  5316.779 | 5314.982 | 1373.889 | 3.869
256_64x64^3  |  9380.338 | 9380.338 | 2342.636 | 4.004
128_128x128^3  |  10779.005 | 10779.005 | 4624.590 | 2.331

<b> KMM Double OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^6  |  821.358 | 1586.475 | 253.348 | 6.262
256_16x16^5  |  1590.090 | 2382.887 | 661.463 | 3.602
256_32x32^4  |  2496.278 | 2496.241 | 1529.754 | 1.632
256_64x64^3  |  4611.845 | 4611.845 | 2993.204 | 1.541
128_128x128^3  |  5935.151 | 5935.151 | 4407.493 | 1.2

<b> KMM Double OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^6  |  802.816 | 1530.923 | 196.505 | 7.791
256_16x16^5  |  1561.879 | 2344.966 | 365.990 | 6.407
256_32x32^4  |  2495.265 | 2494.694 | 911.003 | 2.738
256_64x64^3  |  4267.070 | 4267.070 | 1601.175 | 2.665
128_128x128^3  |  5365.134 | 5365.134 | 2866.481 | 2

## AMD EPYC 7V12 64-Core Processor with AVX2

<b>MKM Float OpX=N and OpF=N</b>

| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
| 256_8x8^5      | 773.198   | 773.198 | 16.941 | 45.641|
| 256_8x8^6      | 516.399   | 516.399 | 33.180 | 15.563|
| 256_16x16^4    | 1399.111 | 1399.111 | 40.418 | 34.616|
| 256_16x16^5    | 601.990  | 601.990 | 56.951 | 10.570|
| 256_32x32^3    | 1800.864 | 1800.864 | 60.281 | 29.875|
| 256_32x32^4    | 1265.890 | 1265.890 | 102.189 | 12.388|
| 256_64x64^2    | 1110.136 | 1110.136 | 85.012 | 13.059|
| 256_64x64^3    | 1430.713 | 1430.713 | 153.630 | 9.313|
| 256_128x128^2  | 2689.599 | 2689.599 | 167.181 | 16.088|
| 128_128x128^3  | 2560.976 | 2560.976 | 240.571 | 10.645|
-----------------------------------

<b> MKM Float OpX=T and OpF=T</b>

| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  63.268 | 63.268 | 36.134 | 1.751
256_8x8^6  |  62.314 | 62.314 | 20.761 | 3.001
256_16x16^4  |  109.803 | 109.803 | 50.452 | 2.176
256_16x16^5  |  84.140 | 84.140 | 31.018 | 2.713
256_32x32^3  |  148.258 | 148.258 | 102.810 | 1.442
256_32x32^4  |  156.784 | 156.784 | 61.492 | 2.550
256_64x64^2  |  720.668 | 720.668 | 208.840 | 3.451
256_64x64^3  |  203.353 | 203.353 | 102.245 | 1.989
256_128x128^2  |  513.992 | 513.992 | 247.513 | 2.077
128_128x128^3  |  394.863 | 394.863 | 126.114 | 3.131
---------------------------------------------

<b> MKM Double OpX=N and OpF=N</b>

| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  404.058 | 404.058 | 18.538 | 21.796
256_8x8^6  |  262.676 | 262.676 | 18.789 | 13.980
256_16x16^4  |  533.115 | 533.115 | 35.370 | 15.072
256_16x16^5  |  294.734 | 294.734 | 34.753 | 8.481
256_32x32^3  |  957.900 | 957.900 | 48.577 | 19.719
256_32x32^4  |  642.908 | 642.908 | 60.983 | 10.542
256_64x64^2  |  520.503 | 520.503 | 40.673 | 12.797
256_64x64^3  |  703.651 | 703.651 | 102.168 | 6.887
256_128x128^2  |  1450.622 | 1450.622 | 111.376 | 13.025
128_128x128^3  |  1149.005 | 1149.005 | 198.005 | 5.803
---------------------

---------------------
<b> KMM Float  OpX=N and OpF=N</b>

| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  453.020 | 453.020 | 32.933 | 13.756
256_8x8^6  |  473.112 | 473.112 | 47.450 | 9.971
256_16x16^4  |  645.404 | 645.404 | 70.036 | 9.215
256_16x16^5  |  455.910 | 455.910 | 89.034 | 5.121
256_32x32^3  |  561.766 | 561.766 | 130.806 | 4.295
256_32x32^4  |  666.560 | 666.560 | 166.682 | 3.999
256_64x64^2  |  548.363 | 548.363 | 205.576 | 2.667
256_64x64^3  |  875.255 | 875.255 | 308.599 | 2.836
256_128x128^2  |  1386.363 | 1386.363 | 402.848 | 3.441
128_128x128^3  |  2075.250 | 2075.250 | 364.437 | 5.694

-----------------------

<b> KMM Float OpX=T and OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  382.543 | 382.543 | 18.812 | 20.335
256_8x8^6  |  397.076 | 397.076 | 33.388 | 11.893
256_16x16^4  |  741.846 | 741.846 | 42.992 | 17.256
256_16x16^5  |  545.828 | 545.828 | 57.080 | 9.562
256_32x32^3  |  499.479 | 499.479 | 70.762 | 7.059
256_32x32^4  |  634.667 | 634.667 | 101.110 | 6.277
256_64x64^2  |  554.877 | 554.877 | 83.266 | 6.664
256_64x64^3  |  1139.939 | 1139.939 | 158.401 | 7.197
256_128x128^2  |  1512.239 | 1512.239 | 167.444 | 9.031
128_128x128^3  |  1563.593 | 1563.593 | 231.462 | 6.755

-----------------------

<b> KMM Double OpX=N and OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  213.357 | 213.357 | 47.517 | 4.490
256_8x8^6  |  242.528 | 242.528 | 34.702 | 6.989
256_16x16^4  |  339.650 | 339.650 | 101.509 | 3.346
256_16x16^5  |  299.177 | 299.177 | 69.257 | 4.320
256_32x32^3  |  259.682 | 259.682 | 165.046 | 1.573
256_32x32^4  |  360.307 | 360.307 | 137.988 | 2.611
256_64x64^2  |  425.382 | 425.382 | 183.031 | 2.324
256_64x64^3  |  524.929 | 524.929 | 264.894 | 1.982
256_128x128^2  |  953.334 | 953.334 | 397.620 | 2.398
128_128x128^3  |  1054.517 | 1054.517 | 515.648 | 2.045

---------------------

<b> KMM Double OpX=T and OpF=T</b>

| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  185.425 | 185.425 | 18.706 | 9.913
256_8x8^6  |  223.118 | 223.118 | 18.999 | 11.744
256_16x16^4  |  218.121 | 218.121 | 36.279 | 6.012
256_16x16^5  |  267.930 | 267.930 | 35.370 | 7.575
256_32x32^3  |  214.034 | 214.034 | 45.268 | 4.728
256_32x32^4  |  343.831 | 343.831 | 62.646 | 5.489
256_64x64^2  |  394.016 | 394.016 | 51.661 | 7.627
256_64x64^3  |  521.852 | 521.852 | 105.072 | 4.967
256_128x128^2  |  708.799 | 708.799 | 105.410 | 6.724
128_128x128^3  |  868.333 | 868.333 | 199.416 | 4.354

## AMD EPYC 9554 64-Core with AVX512

<b> MKM FLOAT  OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  861.987 | 808.113 | 23.740 | 34.040
256_8x8^6  |  1828.224 | 1763.371 | 45.561 | 38.704
256_16x16^4  |  1983.664 | 1858.734 | 60.485 | 30.730
256_16x16^5  |  1535.883 | 1520.168 | 94.904 | 16.018
256_32x32^3  |  1687.114 | 1644.825 | 59.549 | 27.621
256_32x32^4  |  2719.807 | 2662.600 | 155.625 | 17.109
256_64x64^2  |  1095.274 | 1095.274 | 70.786 | 15.473
256_64x64^3  |  3970.696 | 3970.696 | 230.410 | 17.233
256_128x128^2  |  2207.981 | 2207.981 | 146.499 | 15.072
128_128x128^3  |  4118.224 | 4118.224 | 425.253 | 9.684

-------
<b> MKM FLOAT  OpX=T OpF=T</b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  456.993 | 668.545 | 73.985 | 9.036
256_8x8^6  |  1152.851 | 985.269 | 74.670 | 13.195
256_16x16^4  |  1103.094 | 1379.095 | 126.817 | 10.875
256_16x16^5  |  1399.798 | 1396.727 | 255.716 | 5.462
256_32x32^3  |  1657.558 | 1745.752 | 245.143 | 7.121
256_32x32^4  |  2321.623 | 2294.176 | 463.993 | 4.944
256_64x64^2  |  902.076 | 902.076 | 308.838 | 2.921
256_64x64^3  |  3724.143 | 3724.143 | 720.051 | 5.172
256_128x128^2  |  1647.591 | 1647.591 | 636.803 | 2.587
128_128x128^3  |  3877.873 | 3877.873 | 1006.149 | 3.854
16_8x8^8  |  725.972 | 700.658 | 132.661 | 5.282
16_16x16^6  |  934.832 | 900.416 | 255.216 | 3.528
16_32x32^5  |  2200.856 | 2216.109 | 418.041 | 5.301
16_64x64^4  |  3169.570 | 3169.570 | 782.214 | 4.052
16_128x128^3  |  2860.882 | 2860.882 | 789.547 | 3.623

<b> MKM DOUBLE OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  733.332 | 508.927 | 23.690 | 21.483
256_8x8^6  |  719.335 | 649.458 | 33.257 | 19.528
256_16x16^4  |  1462.410 | 1383.907 | 56.292 | 24.584
256_16x16^5  |  735.686 | 729.576 | 61.099 | 11.941
256_32x32^3  |  1314.956 | 1245.896 | 77.133 | 16.153
256_32x32^4  |  1311.112 | 1301.478 | 102.714 | 12.671
256_64x64^2  |  634.011 | 634.011 | 54.221 | 11.693
256_64x64^3  |  1722.227 | 1722.227 | 163.319 | 10.545
256_128x128^2  |  1716.459 | 1716.459 | 143.046 | 11.999
128_128x128^3  |  1888.327 | 1888.327 | 297.792 | 6.341

-------

<b> MKM DOUBLE OpX=T OpF=T</b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  443.956 | 234.263 | 89.587 | 2.615
256_8x8^6  |  438.860 | 491.857 | 85.891 | 5.727
256_16x16^4  |  856.283 | 491.535 | 204.505 | 2.404
256_16x16^5  |  681.703 | 620.554 | 176.323 | 3.519
256_32x32^3  |  1154.914 | 1286.068 | 302.830 | 4.247
256_32x32^4  |  1101.924 | 1095.693 | 328.908 | 3.331
256_64x64^2  |  432.852 | 432.852 | 254.429 | 1.701
256_64x64^3  |  1332.985 | 1332.985 | 628.759 | 2.120
256_128x128^2  |  1605.625 | 1605.625 | 796.604 | 2.016
128_128x128^3  |  1876.462 | 1876.462 | 1021.801 | 1.836
16_8x8^8  |  454.649 | 449.349 | 65.415 | 6.869

-------
<b> KMM FLOAT OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  420.842 | 500.121 | 68.627 | 7.288
256_8x8^6  |  1030.196 | 992.839 | 67.313 | 14.750
256_16x16^4  |  1576.806 | 1271.075 | 132.196 | 9.615
256_16x16^5  |  1287.072 | 1193.908 | 269.336 | 4.433
256_32x32^3  |  1337.324 | 1030.646 | 273.409 | 3.770
256_32x32^4  |  1628.113 | 1404.065 | 471.987 | 2.975
256_64x64^2  |  713.806 | 713.806 | 323.349 | 2.208
256_64x64^3  |  1789.224 | 1789.224 | 691.239 | 2.588
256_128x128^2  |  2329.162 | 2329.162 | 653.122 | 3.566
128_128x128^3  |  4289.726 | 4289.726 | 1014.225 | 4.230

-------
<b> KMM FLOAT OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  405.127 | 497.876 | 28.923 | 17.214
256_8x8^6  |  891.389 | 829.811 | 48.426 | 17.136
256_16x16^4  |  692.173 | 1008.153 | 59.995 | 16.804
256_16x16^5  |  1218.202 | 993.694 | 94.884 | 10.473
256_32x32^3  |  1245.039 | 1019.412 | 62.101 | 16.415
256_32x32^4  |  1733.693 | 1367.983 | 158.412 | 8.636
256_64x64^2  |  681.958 | 681.958 | 73.760 | 9.246
256_64x64^3  |  1786.016 | 1786.016 | 243.329 | 7.340
256_128x128^2  |  2130.730 | 2130.730 | 160.879 | 13.244
128_128x128^3  |  4139.328 | 4139.328 | 428.718 | 9.655

-------
<b> KMM DOUBLE OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  137.334 | 355.449 | 90.209 | 3.940
256_8x8^6  |  588.442 | 602.507 | 86.103 | 6.997
256_16x16^4  |  331.979 | 449.838 | 206.878 | 2.174
256_16x16^5  |  697.520 | 599.861 | 169.456 | 3.540
256_32x32^3  |  582.063 | 649.579 | 303.006 | 2.144
256_32x32^4  |  907.121 | 890.380 | 328.768 | 2.708
256_64x64^2  |  586.203 | 586.203 | 257.631 | 2.275
256_64x64^3  |  1373.891 | 1373.891 | 634.591 | 2.165
256_128x128^2  |  1649.482 | 1649.482 | 824.790 | 2.000
128_128x128^3  |  2062.284 | 2062.284 | 1023.296 | 2.015

-------

<b>KMM DOUBLE OpX=T OpF=T</b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^5  |  128.451 | 334.598 | 23.871 | 14.017
256_8x8^6  |  685.020 | 449.745 | 32.487 | 13.844
256_16x16^4  |  991.336 | 757.621 | 56.133 | 13.497
256_16x16^5  |  807.151 | 681.131 | 61.376 | 11.098
256_32x32^3  |  745.210 | 662.348 | 76.801 | 8.624
256_32x32^4  |  880.639 | 862.566 | 101.946 | 8.461
256_64x64^2  |  553.689 | 553.689 | 53.693 | 10.312
256_64x64^3  |  1368.693 | 1368.693 | 158.982 | 8.609
256_128x128^2  |  1654.926 | 1654.926 | 142.911 | 11.580
128_128x128^3  |  1942.861 | 1942.861 | 302.616 | 6.420