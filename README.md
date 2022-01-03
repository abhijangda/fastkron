
# Current Baseline Build Instructions

1. scipy-sparse
	python3 python/lib_scipy.py
	Sparse scipy 0.0012135505676269531
 
3. numpy
	python3 python/lib_numpy.py
	Time taken numpy 0.9677472114562988
 
4. torch 
	export PYTHONPATH=`pytorch latest directory path`
	Torch time 0.0006659030914306641

5. cublas + saxpy 
	nvcc -lcublas saxpy.cpp && ./a.out 100 100
	Time taken by saxpy: 2 seconds
		
