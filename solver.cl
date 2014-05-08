
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
	__global int* best,
	__constant int* inverses
	)
{
	int id = get_global_id(0);
	__global int* my_output = output + V * id;
	long seed = ((id + 0xCAFE003L) * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);

	__private int guesses[V];
	__private int v1_vals[P];
	__private int v2_vals[P];
	__private int v3_vals[P];

	int num_satisfied;
	int i, j, iter, satisfied, negative;
	int v1, v2, v3, is_b, is_d, coef, eqls, soln;
	int new_best, bestv_v1, bestv_v2, bestv_v3, bestc_v1, bestc_v2, bestc_v3, v1_best;
	int v1_cur, v2_cur, v3_cur, best_v, bestv_v;
	for (i=0; i<V; i++){
		guesses[i] = abs((int)((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1))) % P;
		seed++;
	}
	for (i=0; i<V; i++){
		my_output[i] = guesses[i];
	}
	for (i=0; i<P; i++){
		v1_vals[i] = 0;
		v2_vals[i] = 0;
	}

	best[id] = 0;

	equation_t equation;
	for (iter=0; iter<1500; iter++){
		num_satisfied = 0;
		v1 = abs((int)((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1))) % V;
		seed++;
		v2 = abs((int)((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1))) % V;
		seed++;
		v3 = abs((int)((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1))) % V;
		seed++;
		for (i=0; i<E; i++){
			equation = equations[i];
			satisfied = ((equation.a*guesses[equation.b] + equation.c*guesses[equation.d] + equation.e) % P == 0);
			num_satisfied += satisfied;
			
			is_b = (int)(v1 == equation.b);
			is_d = (int)(v1 == equation.d);
			coef = is_b * equation.a + is_d * equation.c;
			eqls = (0 - equation.e - (1-is_b) * guesses[equation.b] * equation.a - 
							(1-is_d) * guesses[equation.d] * equation.c) % P;
			soln = (inverses[coef] * eqls) % P;
			negative = (int)(soln < 0);
			soln = soln + negative * P;
			// if ((equation.a * (guesses[equation.b] * (1-is_b) + soln * is_b) + 
			// 	equation.c * (guesses[equation.d] * (1-is_d) + soln * is_d) +
			// 	equation.e) % P != 0 && (is_b || is_d)){
			// 	best[id] = 99000000 + i;
			// 	my_output[0] = v1;
			// 	my_output[1] = guesses[equation.b];
			// 	my_output[2] = guesses[equation.d];
			// 	my_output[3] = coef;
			// 	my_output[4] = inverses[coef];
			// 	my_output[5] = -5;
			// 	my_output[6] = is_b;
			// 	my_output[7] = is_d;
			// 	my_output[8] = equation.b;
			// 	my_output[9] = equation.d;
			// 	my_output[10] = -10;
			// 	my_output[11] = soln;
			// 	my_output[12] = eqls;
			// 	my_output[13] = negative;
			// 	my_output[14] = inverses[coef] * eqls;
			// 	my_output[15] = -15;
			// 	my_output[17] = -1;
			// 	return;
			// }
			v1_vals[soln] += (int)(is_b || is_d);

			is_b = (int)(v2 == equation.b);
			is_d = (int)(v2 == equation.d);
			coef = is_b * equation.a + is_d * equation.c;
			eqls = (0 - equation.e - (1-is_b) * guesses[equation.b] * equation.a - 
							(1-is_d) * guesses[equation.d] * equation.c) % P;
			soln = (inverses[coef] * eqls) % P;
			negative = (int)(soln < 0);
			soln = soln + negative * P;
			v2_vals[soln] += (int)(is_b || is_d);

			is_b = (int)(v3 == equation.b);
			is_d = (int)(v3 == equation.d);
			coef = is_b * equation.a + is_d * equation.c;
			eqls = (0 - equation.e - (1-is_b) * guesses[equation.b] * equation.a - 
							(1-is_d) * guesses[equation.d] * equation.c) % P;
			soln = (inverses[coef] * eqls) % P;
			negative = (int)(soln < 0);
			soln = soln + negative * P;
			v3_vals[soln] += (int)(is_b || is_d);
		}
		// if (iter != 0 && num_satisfied < max(bestc_v1, bestc_v2)){
		// 	best[id] = 90000000 + num_satisfied;
		// 	for (i=0; i<V; i++){
		// 		my_output[i] = guesses[i];
		// 	}
		// 	my_output[0] = bestv_v;
		// 	my_output[1] = best_v;
		// 	my_output[2] = max(bestc_v2, bestc_v1);
		// 	my_output[3] = -3;
		// 	return;
		// }

		best[id] = max(best[id], num_satisfied);
		if (best[id] == num_satisfied){
			for (i=0; i<V; i++){
				my_output[i] = guesses[i];
			}
		}

		bestc_v1 = 0;
		bestc_v2 = 0;
		bestc_v3 = 0;
		v1_cur = v1_vals[guesses[v1]];
		v2_cur = v2_vals[guesses[v2]];
		v3_cur = v3_vals[guesses[v3]];
		for (j=0; j<P; j++){
			new_best = (int)(v1_vals[j] > bestc_v1);
			bestc_v1 = max(v1_vals[j], bestc_v1);
			bestv_v1 = new_best ? j : bestv_v1;
			v1_vals[j] = 0;

			new_best = (int)(v2_vals[j] > bestc_v2);
			bestc_v2 = max(v2_vals[j], bestc_v2);
			bestv_v2 = new_best ? j : bestv_v2;
			v2_vals[j] = 0;

			new_best = (int)(v3_vals[j] > bestc_v3);
			bestc_v3 = max(v3_vals[j], bestc_v3);
			bestv_v3 = new_best ? j : bestv_v3;
			v3_vals[j] = 0;
		}
		bestc_v1 = bestc_v1 - v1_cur;
		bestc_v2 = bestc_v2 - v2_cur;
		bestc_v3 = bestc_v3 - v3_cur;

		v1_best = (int)(bestc_v1 > bestc_v2);
		best_v = v1_best ? v1 : v2;
		bestv_v = v1_best ? bestv_v1 : bestv_v2;
		v1_best = (int)(max(bestc_v1, bestc_v2) > bestc_v3);
		best_v = v1_best ? best_v : v3;
		bestv_v = v1_best ? bestv_v : bestv_v3;

		if (bestv_v > 0){
			guesses[best_v] = bestv_v;
		}
	}
}













