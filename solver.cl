
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define V 144
#define E 8096
#define P 1459

typedef struct
{
    int a;
    int b;
    int c;
    int d;
    int e;
} equation_t;


__kernel void solve(
	__global equation_t* equations,
	__global int* output,
	volatile __global int* best,
	volatile __global int* lock,
	__constant int* inverses
	)
{
	int id = get_global_id(0);
	long seed = ((id + 0xDEADFACEL) * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);

	int guesses[V];
	int v1_vals[P];
	int v2_vals[P];

	int num_satisfied;
	int i, j, iter, satisfied;
	int v1, v2, is_b, is_d, coef, eqls, soln;
	int new_best, bestv_v1, bestv_v2, bestc_v1, bestc_v2, v1_best;
	for (i=0; i<V; i++){
		guesses[i] = abs((int)((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1))) % P;
		seed++;
	}
	for (i=0; i<P; i++){
		v1_vals[i] = 0;
		v2_vals[i] = 0;
	}

	int prev_best;
	int waiting;
	*best = 0;
	*lock = 0;

	barrier(CLK_GLOBAL_MEM_FENCE);

	equation_t equation;
	for (iter=0; iter<10; iter++){
		num_satisfied = 0;
		v1 = abs((int)((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1))) % V;
		seed++;
		v2 = abs((int)((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1))) % V;
		seed++;
		for (i=0; i<E; i++){
			equation = equations[i];
			satisfied = (equation.a*guesses[equation.b] + equation.c*guesses[equation.d] + equation.e == 0);
			num_satisfied += satisfied;
			
			is_b = (int)(v1 == equation.b);
			is_d = (int)(v1 == equation.d);
			coef = is_b * equation.a + is_d * equation.c;
			eqls = 0 - equation.e - (1-is_b) * guesses[equation.b] * equation.a - 
							(1-is_d) * guesses[equation.d] * equation.c;
			soln = (inverses[coef] * eqls) % P + P;
			soln = soln % P;
			v1_vals[soln]++;

			is_b = (int)(v2 == equation.b);
			is_d = (int)(v2 == equation.d);
			coef = is_b * equation.a + is_d * equation.c;
			eqls = 0 - equation.e - (1-is_b) * guesses[equation.b] * equation.a - 
							(1-is_d) * guesses[equation.d] * equation.c;
			soln = (inverses[coef] * eqls) % P + P;
			soln = soln % P;
			v2_vals[soln]++;
		}

		prev_best = atomic_max(best, num_satisfied);
		if (prev_best == num_satisfied){
			waiting = 1;
			int timeout = 4000000;
			while (waiting && timeout>0){
				if (!atomic_xchg(lock, 1)){
					for (i=0; i<V; i++){
						output[i] = guesses[i];
					}
					atomic_xchg(lock, 0);
					waiting = 0;
				}

				timeout--;
				if (timeout == 0){
					atomic_max(best, 9999999);
				}
			}
		}

		bestc_v1 = 0;
		bestc_v2 = 0;
		for (j=0; j<P; j++){
			new_best = (int)(v1_vals[j] > bestc_v1);
			bestc_v1 = max(v1_vals[j], bestc_v1);
			bestv_v1 = new_best ? j : bestv_v1;

			new_best = (int)(v2_vals[j] > bestc_v2);
			bestc_v2 = max(v2_vals[j], bestc_v2);
			bestv_v2 = new_best ? j : bestv_v2;
		}
		v1_best = (int)(bestc_v1 > bestc_v2);
		v1 = v1_best ? v1 : v2;
		soln = v1_best ? bestv_v1 : bestv_v2;
		guesses[v1] = soln;
	}
}













