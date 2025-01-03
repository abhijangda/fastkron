Below tables show the throughput in Tera FLOPs achieved by FastKron without fusion, FastKron, GPyTorch (using Shuffle Algorithm), and speedup.
First column shows the shape of KMM/MKM in the format M_PxQ^N, such that, M is the number of rows of $X$ in MKM or columns of $X$ in KMM, PxQ is the size of each Kronecker Factor and N is the number of Kronecker Factors.
The last column of speedup shows that FastKron performs orders of magnitude faster than GPyTorch on different sizes, float and double data types, operation on X and Factors (N for no-transpose and T for transpose), when running on wide range of CPUs and GPUs.

## NVIDIA A100 80GB
<b> MKM FLOAT OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  3409.436 | 5852.703 | 763.094 | 7.670
1024_8x8^6  |  3553.610 | 6157.745 | 646.565 | 9.524
1024_16x16^4  |  6918.694 | 6880.698 | 1465.086 | 4.696
1024_16x16^5  |  7046.385 | 7046.998 | 1440.325 | 4.893
1024_32x32^3  |  12706.360 | 12703.822 | 3671.812 | 3.460
1024_32x32^4  |  13692.792 | 13692.722 | 2741.848 | 4.994
1024_64x64^2  |  13127.553 | 13127.553 | 3740.531 | 3.510
1024_64x64^3  |  16948.776 | 16948.776 | 4473.117 | 3.789
1024_128x128^2  |  16329.991 | 16329.991 | 9499.762 | 1.719
320_128x128^3  |  17311.574 | 17311.574 | 7719.255 | 2.243
16_8x8^8  |  3511.135 | 5957.818 | 887.348 | 6.714
16_16x16^6  |  7004.467 | 7005.046 | 2264.016 | 3.094
16_32x32^5  |  13567.418 | 13568.617 | 4906.991 | 2.765
16_64x64^4  |  16933.984 | 16933.984 | 8748.702 | 1.936
16_128x128^3  |  16806.341 | 16806.341 | 11212.979 | 1.499

--------------------
<b> MKM FLOAT  OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  3302.982 | 4805.036 | 930.603 | 5.163
1024_8x8^6  |  3465.595 | 5259.087 | 959.237 | 5.483
1024_16x16^4  |  6561.171 | 6568.759 | 2577.462 | 2.549
1024_16x16^5  |  6805.934 | 6803.972 | 2608.306 | 2.609
1024_32x32^3  |  11774.388 | 11765.671 | 5654.947 | 2.081
1024_32x32^4  |  12616.478 | 12616.949 | 5863.293 | 2.152
1024_64x64^2  |  10411.507 | 10411.507 | 4055.835 | 2.567
1024_64x64^3  |  13976.185 | 13976.185 | 10263.640 | 1.362
1024_128x128^2  |  13801.010 | 13801.010 | 12217.293 | 1.130
320_128x128^3  |  14751.687 | 14751.687 | 12922.255 | 1.142
16_8x8^8  |  3502.330 | 5364.954 | 926.063 | 5.793
16_16x16^6  |  6939.095 | 6939.758 | 2443.735 | 2.840
16_32x32^5  |  13034.639 | 13035.847 | 5431.185 | 2.400
16_64x64^4  |  14592.892 | 14592.892 | 9748.348 | 1.497
16_128x128^3  |  14632.814 | 14632.814 | 12270.274 | 1.193

-------------------------
<b> MKM DOUBLE OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  1745.470 | 3356.203 | 386.201 | 8.690
1024_8x8^6  |  1766.724 | 3643.161 | 381.557 | 9.548
1024_16x16^4  |  3494.537 | 5004.218 | 907.097 | 5.517
1024_16x16^5  |  3481.699 | 4639.286 | 965.840 | 4.803
1024_32x32^3  |  6559.686 | 6560.363 | 1956.726 | 3.353
1024_32x32^4  |  6763.349 | 6763.214 | 1973.439 | 3.427
1024_64x64^2  |  6428.029 | 6428.029 | 3427.397 | 1.875
1024_64x64^3  |  8082.399 | 8082.399 | 3406.866 | 2.372
1024_128x128^2  |  7745.218 | 7745.218 | 6967.741 | 1.112
320_128x128^3  |  8196.844 | 8196.844 | 7195.432 | 1.139
16_8x8^8  |  1740.157 | 3431.515 | 455.711 | 7.530
16_16x16^6  |  3420.404 | 4833.526 | 1273.347 | 3.796
16_32x32^5  |  6621.797 | 6622.707 | 2852.013 | 2.322
16_64x64^4  |  8060.372 | 8060.372 | 5372.391 | 1.500
16_128x128^3  |  7953.398 | 7953.398 | 8607.797 | 0.924

---------------------------------
<b>  MKM DOUBLE OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  1688.329 | 2549.555 | 457.646 | 5.571
1024_8x8^6  |  1718.586 | 2841.198 | 462.281 | 6.146
1024_16x16^4  |  3262.245 | 3724.450 | 1311.550 | 2.840
1024_16x16^5  |  3310.383 | 4231.816 | 1329.253 | 3.184
1024_32x32^3  |  4209.371 | 4211.044 | 3034.566 | 1.388
1024_32x32^4  |  4717.311 | 4717.048 | 3077.022 | 1.533
1024_64x64^2  |  3971.319 | 3971.319 | 3827.639 | 1.038
1024_64x64^3  |  4875.688 | 4875.688 | 5473.699 | 0.891
1024_128x128^2  |  5552.433 | 5552.433 | 8614.795 | 0.645
320_128x128^3  |  6133.317 | 6133.317 | 9009.684 | 0.681
16_8x8^8  |  1727.876 | 2940.330 | 455.797 | 6.451
16_16x16^6  |  3356.960 | 4148.897 | 1301.352 | 3.188
16_32x32^5  |  5056.602 | 5056.588 | 2988.523 | 1.692
16_64x64^4  |  5098.110 | 5098.110 | 5304.154 | 0.961
16_128x128^3  |  6111.952 | 6111.952 | 9077.245 | 0.673

------------------------
<b> KMM FLOAT OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  3209.096 | 3712.318 | 935.133 | 3.970
1024_8x8^6  |  3328.425 | 3893.090 | 959.956 | 4.055
1024_16x16^4  |  5859.074 | 7333.173 | 2588.053 | 2.833
1024_16x16^5  |  5942.001 | 7229.497 | 2608.988 | 2.771
1024_32x32^3  |  10798.356 | 10789.192 | 5705.815 | 1.891
1024_32x32^4  |  11368.397 | 11358.231 | 5864.498 | 1.937
1024_64x64^2  |  12535.800 | 12535.800 | 5084.217 | 2.466
1024_64x64^3  |  17626.656 | 17626.656 | 10274.225 | 1.716
1024_128x128^2  |  16738.971 | 16738.971 | 12402.340 | 1.350
320_128x128^3  |  17924.391 | 17924.391 | 12925.577 | 1.387
16_8x8^8  |  3425.617 | 4270.302 | 926.611 | 4.609
16_16x16^6  |  6630.241 | 8280.568 | 2445.941 | 3.385
16_32x32^5  |  12110.266 | 12100.717 | 5432.077 | 2.228
16_64x64^4  |  15599.293 | 15599.293 | 9758.678 | 1.599
16_128x128^3  |  16822.028 | 16822.028 | 12334.405 | 1.364

--------------------------
<b>  KMM FLOAT OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  3155.209 | 3894.732 | 766.805 | 5.079
1024_8x8^6  |  3275.254 | 3957.744 | 646.851 | 6.118
1024_16x16^4  |  5433.123 | 5966.163 | 1468.370 | 4.063
1024_16x16^5  |  4959.528 | 6136.254 | 1440.424 | 4.260
1024_32x32^3  |  9001.688 | 9006.789 | 3697.840 | 2.436
1024_32x32^4  |  9685.644 | 9697.865 | 2742.265 | 3.536
1024_64x64^2  |  11424.519 | 11424.519 | 4718.776 | 2.421
1024_64x64^3  |  15598.997 | 15598.997 | 4476.007 | 3.485
1024_128x128^2  |  14225.298 | 14225.298 | 9612.806 | 1.480
320_128x128^3  |  15721.111 | 15721.111 | 7721.869 | 2.036
16_8x8^8  |  3316.761 | 3916.104 | 887.774 | 4.411
16_16x16^6  |  6214.904 | 7134.071 | 2265.549 | 3.149
16_32x32^5  |  10056.396 | 10054.779 | 4908.596 | 2.048
16_64x64^4  |  14120.498 | 14120.498 | 8757.634 | 1.612
16_128x128^3  |  13626.052 | 13626.052 | 11271.210 | 1.209

-------------------------
<b> KMM DOUBLE  OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  1650.882 | 2010.758 | 458.701 | 4.384
1024_8x8^6  |  1698.419 | 2114.674 | 462.432 | 4.573
1024_16x16^4  |  3108.919 | 4058.923 | 1314.833 | 3.087
1024_16x16^5  |  3154.104 | 3868.262 | 1329.297 | 2.910
1024_32x32^3  |  5874.138 | 5864.384 | 3055.843 | 1.919
1024_32x32^4  |  6112.378 | 6115.390 | 3077.656 | 1.987
1024_64x64^2  |  6271.620 | 6271.620 | 4272.865 | 1.468
1024_64x64^3  |  7802.807 | 7802.807 | 5472.646 | 1.426
1024_128x128^2  |  7409.269 | 7409.269 | 8700.506 | 0.852
320_128x128^3  |  7726.575 | 7726.575 | 9011.382 | 0.857
16_8x8^8  |  1757.895 | 2175.358 | 455.911 | 4.771
16_16x16^6  |  3351.408 | 4292.292 | 1301.791 | 3.297
16_32x32^5  |  6397.391 | 6395.379 | 2988.513 | 2.140
16_64x64^4  |  7768.573 | 7768.573 | 5308.834 | 1.463
16_128x128^3  |  7693.426 | 7693.426 | 9110.144 | 0.844

-----------------------
<b> KMM DOUBLE OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
1024_8x8^5  |  1588.423 | 1913.256 | 386.960 | 4.944
1024_8x8^6  |  1640.095 | 2028.033 | 381.677 | 5.313
1024_16x16^4  |  3032.454 | 3956.202 | 908.671 | 4.354
1024_16x16^5  |  3096.742 | 3800.668 | 966.004 | 3.934
1024_32x32^3  |  5196.593 | 5201.270 | 1964.121 | 2.648
1024_32x32^4  |  5489.904 | 5492.423 | 1973.784 | 2.783
1024_64x64^2  |  5597.288 | 5597.288 | 3868.407 | 1.447
1024_64x64^3  |  6819.719 | 6819.719 | 3407.945 | 2.001
1024_128x128^2  |  6343.786 | 6343.786 | 7028.364 | 0.903
320_128x128^3  |  6782.259 | 6782.259 | 7196.519 | 0.942
16_8x8^8  |  1705.351 | 2095.244 | 455.787 | 4.597
16_16x16^6  |  3281.303 | 4162.004 | 1273.764 | 3.267
16_32x32^5  |  5790.853 | 5790.912 | 2852.693 | 2.030
16_64x64^4  |  7073.740 | 7073.740 | 5378.888 | 1.315
16_128x128^3  |  6730.337 | 6730.337 | 8635.995 | 0.779

## NVIDIA V100 16GB

<b> MKM Float OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^6  |  1683.405 | 3751.038 | 375.437 | 9.991
256_16x16^5  |  3346.651 | 5450.749 | 755.190 | 7.218
256_32x32^4  |  6563.776 | 8758.502 | 1372.864 | 6.380
256_64x64^3  |  10445.493 | 10445.493 | 2327.181 | 4.488
128_128x128^3  |  11277.279 | 11277.279 | 4593.815 | 2.455

<b> MKM Float  OpX=T OpF=T</b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^6  |  1666.631 | 2941.795 | 620.949 | 4.738
256_16x16^5  |  3139.310 | 4855.850 | 1504.183 | 3.228
256_32x32^4  |  6354.887 | 7158.841 | 3140.264 | 2.280
256_64x64^3  |  9201.548 | 9201.548 | 6090.472 | 1.511
128_128x128^3  |  9163.544 | 9163.544 | 8792.515 | 1.042

<b> MKM Double OpX=N OpF=N </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^6  |  846.328 | 2310.013 | 196.134 | 11.778
256_16x16^5  |  1657.223 | 2395.805 | 364.473 | 6.573
256_32x32^4  |  2657.074 | 4330.767 | 913.510 | 4.741
256_64x64^3  |  5272.823 | 5272.823 | 1599.346 | 3.297
128_128x128^3  |  5436.299 | 5436.299 | 2871.772 | 1.893

<b> MKM Double OpX=T OpF=T </b>
| Shape          | FK-no-fuse|FastKron| GPyTorch|Speedup|
|----------------|-----------|--------|---------|-------|
256_8x8^6  |  834.441 | 1594.787 | 252.848 | 6.307
256_16x16^5  |  1637.819 | 2339.475 | 665.392 | 3.516
256_32x32^4  |  2629.389 | 3109.499 | 1526.620 | 2.037
256_64x64^3  |  3903.782 | 3903.782 | 2988.321 | 1.306
128_128x128^3  |  4334.606 | 4334.606 | 4411.487 | 1.1

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