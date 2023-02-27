// nvcc -arch=compute_70 -code=sm_70 giscup7.cu fdkernel2.cu fdlib2.cpp utility.cpp -O2 -o giscup7g   --ptxas-options=-v
//nvcc -arch=compute_70 -code=sm_70 giscup7.cu fdkernel2.cu fdlib2.cpp utility.cpp -O2 -o giscup7g
//nvcc -arch=compute_75 -code=sm_75 giscup7.cu fdkernel2.cu fdlib2.cpp utility.cpp -O2 -o giscup7g 

#include <stdio.h>
#include <algorithm>
#include <cassert>
#include "utility.h"
#include "fdlib2.h"
#include "fdkernel3.cuh"

#define DEBUG 0
#define FAC 100000

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );	
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



  int main(int argc,char** argv)
  {
    timeval t0,t1;
    
    gettimeofday(&t0, NULL);
    double *ux=NULL,*uy=NULL;
    uint32_t *ll=NULL;
    size_t num_x=read_field("../data/new-data-HWY_20_AND_LOCUST_x.bin",ux);
    size_t num_y=read_field("../data/new-data-HWY_20_AND_LOCUST_y.bin",uy);
    assert(num_x==num_y);
    size_t num_t=read_field("../data/new-data-HWY_20_AND_LOCUST_l.bin",ll);
    gettimeofday(&t1, NULL);
    calc_time("file read time in ms",t0,t1);
    
    const uint32_t maxl=*std::max_element(ll,ll+num_t);
    const double minx=*std::min_element(ux,ux+num_x);
    const double maxx=*std::max_element(ux,ux+num_x);
    const double miny=*std::min_element(uy,uy+num_y);
    const double maxy=*std::max_element(uy,uy+num_y);
    printf("%20.15f %20.15f %20.15f %20.15f\n",minx,miny,maxx,maxy);
    
    //double * xx=ux;
    //double * yy=uy;
        
    float *xx=new float[num_x];
    float *yy=new float[num_y];
    assert(xx!=NULL && yy!=NULL); 
    
    for(int i=0;i<num_x;i++)
       xx[i]=(ux[i]-minx)*FAC;
    for(int i=0;i<num_y;i++)
       yy[i]=(uy[i]-miny)*FAC;
   
    delete [] ux;
    delete [] uy;
 
    uint32_t *pp=new uint32_t[num_t];
     assert(pp!=NULL);
 
     pp[0]=0;
     for(int i=1;i<num_t;i++)
        pp[i]=pp[i-1]+ll[i-1];
        
     size_t nt=num_t;
     size_t num_p=0;
     
     for(size_t i=0;i<nt;i++)
        for(size_t j=0;j<nt;j++)
            if(ll[i]<=ll[j]) num_p++;
    assert(num_p>0);
    
    printf("total num_p=%zu\n",num_p);

    uint32_t *fids=new uint32_t[num_p];
    uint32_t *tids=new uint32_t[num_p]; 
    assert(fids!=NULL && tids!=NULL); 
    
    size_t p=0;
    for(size_t i=0;i<nt;i++)
        for(size_t j=0;j<nt;j++)
        {
	    if(ll[i]<=ll[j]) 
	    {
	        fids[p]=i;
	        tids[p]=j;
	        p++;
	     }   
	}
    assert(p==num_p);
    printf("use num_p=%zu\n",num_p);

    gettimeofday(&t0, NULL);
    calc_time("prep time in ms",t1,t0);
           
    size_t tot_mem=0;
    //run gpu code
    float *d_xx=NULL,*d_yy=NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&d_xx,num_x* sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&d_yy,num_y* sizeof(float)));
    assert(d_xx!=NULL && d_yy!=NULL);
    HANDLE_ERROR( cudaMemcpy( d_xx, xx, num_x * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( d_yy, yy, num_y * sizeof(float), cudaMemcpyHostToDevice ) );    
    tot_mem+=num_x * sizeof(float)+num_y * sizeof(float);
    
    uint32_t *d_ll=NULL,*d_pp=NULL;    
    HANDLE_ERROR( cudaMalloc( (void**)&d_pp,num_t* sizeof(uint32_t)));
    HANDLE_ERROR( cudaMalloc( (void**)&d_ll,num_t* sizeof(uint32_t)));
    assert(d_pp!=NULL && d_ll!=NULL);
    
    HANDLE_ERROR( cudaMemcpy( d_pp, pp, num_t * sizeof(uint32_t), cudaMemcpyHostToDevice ) );  
    HANDLE_ERROR( cudaMemcpy( d_ll, ll, num_t * sizeof(uint32_t), cudaMemcpyHostToDevice ) );   
    tot_mem+=num_t * sizeof(uint32_t)+num_t * sizeof(uint32_t);

    uint32_t *d_fids=NULL,*d_tids=NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&d_fids,num_p* sizeof(uint32_t)));
    HANDLE_ERROR( cudaMalloc( (void**)&d_tids,num_p* sizeof(uint32_t)));

    assert(d_fids!=NULL && d_tids!=NULL);
    HANDLE_ERROR( cudaMemcpy( d_fids, fids, num_p * sizeof(uint32_t), cudaMemcpyHostToDevice ) );    
    HANDLE_ERROR( cudaMemcpy( d_tids, tids, num_p * sizeof(uint32_t), cudaMemcpyHostToDevice ) );       
    tot_mem+=num_p * sizeof(uint32_t)+num_p * sizeof(uint32_t);
    
    /*uint32_t *d_op=NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&d_op,num_p* sizeof(uint32_t)));
    HANDLE_ERROR( cudaMemcpy( d_op, op, num_p * sizeof(uint32_t), cudaMemcpyHostToDevice ) );       
    tot_mem+=num_p * sizeof(uint32_t)

    float *d_work=NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&d_work,sz*sizeof(float)));
    assert(d_work!=NULL);
    tot_mem+=sz*sizeof(float);    
      
    */
    
    float *d_dis=NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&d_dis,num_p* sizeof(float)));
    assert(d_dis!=NULL);    
    tot_mem+=num_p* sizeof(float);
      
    printf("tot_mem=%zu\n",tot_mem);
    
    gettimeofday(&t0, NULL);
    calc_time("gpu prep time in ms",t1,t0);
 
    const size_t nthreads=((maxl-1)/32+1)*32;
    printf("nthreads=%zu\n",nthreads);
    assert(nthreads<=1024);
        
    //3 times of max of l_n for shared memory size
    fd_kernel_shared_loop<float,128> <<< num_p, nthreads>>> (d_xx,d_yy,d_ll,d_pp,d_fids,d_tids,d_dis);
    
    HANDLE_ERROR( cudaDeviceSynchronize() );

    gettimeofday(&t1, NULL);
    calc_time("gpu kernel time in ms",t0,t1);
    
    float *h_dis=new float[num_p];
    assert(h_dis!=NULL);
    HANDLE_ERROR( cudaMemcpy( h_dis, d_dis, num_p * sizeof(float), cudaMemcpyDeviceToHost ) );
    gettimeofday(&t0, NULL);
    calc_time("GPU-->CPU time in ms",t1,t0);
    
    FILE *fp=fopen("nt_gout.bin","w");
    assert(fp!=NULL);
    fwrite(h_dis,num_p,sizeof(float),fp);
    fclose(fp);
    gettimeofday(&t1, NULL);
    calc_time("file output time in ms",t0,t1);
    
    cudaFree(d_xx);
    cudaFree(d_yy);
    cudaFree(d_ll);
    cudaFree(d_pp);
    cudaFree(d_fids);
    cudaFree(d_tids);
    cudaFree(d_dis);
  
    delete[] xx;
    delete[] yy;
    delete[] ll;
    delete[] pp;
    delete[] fids;
    delete[] tids;
    
    delete[] h_dis;
    
}

