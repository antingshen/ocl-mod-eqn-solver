#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include "clhelp.h"

typedef struct equation_t
{
    int a;
    int b;
    int c;
    int d;
    int e;
};

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

int assign(equation_t* equations, int* output, int E, int V, int P){

    // #pragma OPENCL EXTENSION cl_khr_int64 : require
    // #pragma OPENCL EXTENSION cl_khr_int64 : require
    // #pragma OPENCL EXTENSION cl_khr_int64 : require

    double t0, t1;

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

    cl_mem g_in, g_out, g_inverse, g_best, g_lock;

    g_in = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(equation_t)*E,NULL,&err); CHK_ERR(err); 
    g_out = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(int)*V,NULL,&err); CHK_ERR(err);
    g_best = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(int),NULL,&err); CHK_ERR(err);
    g_lock = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(int),NULL,&err); CHK_ERR(err);
    g_inverse = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
        sizeof(int)*P,NULL,&err); CHK_ERR(err);

    err = clEnqueueWriteBuffer(cv.commands, g_in, true, 0, sizeof(equation_t)*E,
        equations, 0, NULL, NULL); CHK_ERR(err);

    size_t global_work_size[1] = {2};

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
        sizeof(cl_mem), &g_lock); 
        CHK_ERR(err);

    int* inverse = new int[P];
    int i;
    for (i=1; i<P; i++){
        inverse[i] = mod_inv(i, P);
    }
    err = clEnqueueWriteBuffer(cv.commands, g_inverse, true, 0, sizeof(int)*P,
        inverse, 0, NULL, NULL); CHK_ERR(err);
    err = clSetKernelArg(kernel,4,
        sizeof(cl_mem), &g_inverse); CHK_ERR(err);

    printf("Starting GPU...\n");
    t0 = timestamp();
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
    t1 = timestamp();
    printf("GPU work complete in %.4f\n", t1-t0);
    CHK_ERR(err);

    err = clEnqueueReadBuffer(cv.commands, g_out, true, 0, sizeof(int)*V,
        output, 0, NULL, NULL);
    CHK_ERR(err);

    clReleaseMemObject(g_in);
    clReleaseMemObject(g_out);

    uninitialize_ocl(cv);

    delete[] inverse;

    return 1;
}


int main(int argc, char *argv[]){
    int t = 0; // test number
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

    assign(equations, output, E, V, P);

    fprintf(fout, "%d", output[0]);
    for (int i = 1; i < V ;i++)
        fprintf(fout, " %d", output[i]);
    fprintf(fout, "\n");

    delete[] equations;
    delete[] output;

    fclose(fout);
    return 0;
}
