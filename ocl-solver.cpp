#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include "clhelp.h"

typedef struct
{
    int a;
    int b;
    int c;
    int d;
    int e;
} equation_t;

int mod_inv(int x, int p)
{
    int p0 = p, t, q;
    int x0 = 0, x1 = 1;
    if (p == 1) return 1;
    while (x > 1) {
        q = x / p;
        t = p, p = x % p, x = t;
        t = x0, x0 = x1 - q * x0, x1 = t;
    }
    if (x1 < 0) x1 += p0;
    return x1;
}

#define NUM_THREADS 2048
#define NUM_ITERS 3000
int assign(equation_t* equations, int* output, int E, int V, int P){

    // OpenCL setup
    std::string kernel_str;

    std::string name_str = std::string("solve");
    std::string kernel_file = std::string("solver.cl");

    cl_vars_t cv; 
    cl_kernel kernel;

    readFile(kernel_file, kernel_str);

    initialize_ocl(cv);

    compile_ocl_program(kernel, cv, kernel_str.c_str(),
    name_str.c_str());

    cl_int err = CL_SUCCESS;

    cl_mem g_in, g_out, g_inverse, g_best, g_guesses, g_seed;
    int i,j;

    g_in = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(equation_t)*E,NULL,&err); CHK_ERR(err); 
    g_out = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(int)*V*NUM_THREADS,NULL,&err); CHK_ERR(err);
    g_best = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(int)*NUM_THREADS,NULL,&err); CHK_ERR(err);
    g_guesses = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(int)*V*NUM_THREADS,NULL,&err); CHK_ERR(err);
    g_inverse = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(int)*P,NULL,&err); CHK_ERR(err);

    err = clEnqueueWriteBuffer(cv.commands, g_in, true, 0, sizeof(equation_t)*E,
        equations, 0, NULL, NULL); CHK_ERR(err);
    int* outputs = new int[V*NUM_THREADS];
    memset(outputs, 0, sizeof(int)*V*NUM_THREADS);
    err = clEnqueueWriteBuffer(cv.commands, g_out, true, 0, sizeof(int)*V*NUM_THREADS,
        outputs, 0, NULL, NULL); CHK_ERR(err);
    err = clEnqueueWriteBuffer(cv.commands, g_best, true, 0, sizeof(int)*NUM_THREADS,
        outputs, 0, NULL, NULL); CHK_ERR(err);

    int* init_guesses = new int[V*NUM_THREADS];
    for (i=0; i<V*NUM_THREADS; i++){
        init_guesses[i] = rand() % P;
    }
    err = clEnqueueWriteBuffer(cv.commands, g_guesses, true, 0, sizeof(int)*V*NUM_THREADS,
        init_guesses, 0, NULL, NULL); CHK_ERR(err);


    size_t global_work_size[1] = {NUM_THREADS};
    // size_t local_work_size[1] = {1};

    err = clSetKernelArg(kernel,0,
        sizeof(cl_mem), &g_in); 
        CHK_ERR(err);
    err = clSetKernelArg(kernel,1,
        sizeof(cl_mem), &g_out); 
        CHK_ERR(err);
    err = clSetKernelArg(kernel,2,
        sizeof(cl_mem), &g_best); 
        CHK_ERR(err);
    err = clSetKernelArg(kernel,3,
        sizeof(cl_mem), &g_guesses);
        CHK_ERR(err);

    int* inverse = new int[P];
    for (i=1; i<P; i++){
        inverse[i] = mod_inv(i, P);
    }
    err = clEnqueueWriteBuffer(cv.commands, g_inverse, true, 0, sizeof(int)*P,
        inverse, 0, NULL, NULL); CHK_ERR(err);
    err = clSetKernelArg(kernel,4,
        sizeof(cl_mem), &g_inverse); CHK_ERR(err);

    g_seed = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(int),NULL,&err); CHK_ERR(err);
    err = clSetKernelArg(kernel,5,
        sizeof(cl_mem), &g_seed); CHK_ERR(err);

    int rand_int[1];
    // srand(1);
    int* bests = new int[NUM_THREADS];
    for (i=0; i<NUM_ITERS; i++){

        rand_int[0] = rand();
        err = clEnqueueWriteBuffer(cv.commands, g_seed, true, 0, sizeof(int),
            rand_int, 0, NULL, NULL); CHK_ERR(err);

        err = clEnqueueNDRangeKernel(cv.commands, 
            kernel,
            1,//work_dim,
            NULL, //global_work_offset
            global_work_size, //global_work_size
            NULL, //local_work_size
            0, //num_events_in_wait_list
            NULL, //event_wait_list
            NULL //
            );
        CHK_ERR(err);

        // err = clEnqueueReadBuffer(cv.commands, g_guesses, true, 0, sizeof(int)*V*NUM_THREADS,
        //     outputs, 0, NULL, NULL);
        // CHK_ERR(err); int j;
        // for (j=0; j<V; j++){
        //     if (outputs[j] != init_guesses[j]){
        //         printf("%d:%d ",j,outputs[j]);
        //     }
        // } printf("\n");


        err = clEnqueueReadBuffer(cv.commands, g_best, true, 0, sizeof(int)*NUM_THREADS,
            bests, 0, NULL, NULL);
        CHK_ERR(err);

        int best_t = 0;
        int best_v = 0;
        for (j=0; j<NUM_THREADS; j++){
            // printf("%d ", bests[j]);
            if (bests[j] > best_v){
                best_v = bests[j];
                best_t = j;
            }
        } // printf("\n");
        int* best_output = outputs + best_t * V;
        for (j=0; j<V; j++){
            output[j] = best_output[j];
        }

        printf("%d of %d, best: %d\n", i+1, NUM_ITERS, best_v);

        if (interrupted) {
            break;
        }
    }

    err = clEnqueueReadBuffer(cv.commands, g_out, true, 0, sizeof(int)*V*NUM_THREADS,
        outputs, 0, NULL, NULL);
    CHK_ERR(err);
    err = clEnqueueReadBuffer(cv.commands, g_best, true, 0, sizeof(int)*NUM_THREADS,
        bests, 0, NULL, NULL);
    CHK_ERR(err);

    int best_t = 0;
    int best_v = 0;
    for (i=0; i<NUM_THREADS; i++){
        // printf("%d ", bests[i]);
        if (bests[i] > best_v){
            best_v = bests[i];
            best_t = i;
        }
    } // printf("\n");
    int* best_output = outputs + best_t * V;
    for (i=0; i<V; i++){
        output[i] = best_output[i];
    }

    printf("Outputs read. Best: %d\n", best_v);    
    delete[] bests;
    delete[] outputs;
    delete[] init_guesses;

    clReleaseMemObject(g_in);
    clReleaseMemObject(g_out);
    clReleaseMemObject(g_inverse);
    clReleaseMemObject(g_guesses);
    clReleaseMemObject(g_best);
    clReleaseMemObject(g_seed);

    uninitialize_ocl(cv);

    delete[] inverse;

    return 1;
}


int main(int argc, char *argv[]){
    int t = 22; // test number
    FILE * fout = fopen ("answer.out", "w");

    char filename[10];
    sprintf (filename, "%d.in", t);
    FILE * fin = fopen(filename, "r");

    int V, E, P;
    fscanf(fin, "%d%d%d", &V, &E, &P);

    equation_t* equations = new equation_t[E];

    for (int i = 0; i < E; i++) {
        int a, b, c, d, e;
        fscanf(fin, "%d%d%d%d%d", &a, &b, &c, &d, &e);
        equations[i].a = (int)a;
        equations[i].b = (int)b;
        equations[i].c = (int)c;
        equations[i].d = (int)d;
        equations[i].e = (int)e;
    }
    int* output = new int[V];
    memset (output, 0, sizeof(output));

    double t0, t1;
    t0 = timestamp();
    assign(equations, output, E, V, P);
    t1 = timestamp();
    printf("Total seconds elapsed: %.6f\n", t1-t0);

    fprintf(fout, "%d", output[0]);
    for (int i = 1; i < V ;i++)
        fprintf(fout, " %d", output[i]);
    fprintf(fout, "\n");

    delete[] equations;
    delete[] output;

    fclose(fout);
    return 0;
}
