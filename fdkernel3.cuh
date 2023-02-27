#ifndef FDKERNEL3_CUH
#define FDKERNEL3_CUH

#include <stdint.h>

//#define CUDEBUG blockIdx.x==1 && threadIdx.x==0
#define CUDEBUG 0

template <class T>
__forceinline__ __device__ T  dist_sqr(const std::size_t i,const std::size_t j,
    const T *xx1,const T *yy1,const T *xx2,const T *yy2)
{
    T fd=(xx1[i]-xx2[j])*(xx1[i]-xx2[j])+(yy1[i]-yy2[j])*(yy1[i]-yy2[j]);
    return(fd);
}

template <class T>
__forceinline__ __device__ T dmax(const T *a1,const T *a2,const uint32_t m,const T t3 )
{
    T d1=a2[m]; //(s-1,t)
    T d2=a1[m]; //(s-1,t-1)
    T d3=a2[m+1];//(s,t-1)
    
    T t1=(d1<d2)?d1:d2;
    T t2=(t1<d3)?t1:d3;
    T d=(t3>t2)?t3:t2;
if(CUDEBUG)
{
    printf("dmax: m=%d d1=%10.5f d2=%10.5f d3=%10.5f t3=%10.5f d=%10.5f\n",m,d1,d2,d3,t3,d);
}
    return(d);

}

#endif

template <class T, size_t B>
__global__ void fd_kernel_shared_loop(const T *xx,const T *yy,
    const uint32_t * ll,const uint32_t * pp,
    const uint32_t * fids,const uint32_t *tids,T *dis)
    
{
    __shared__ T base[B*3];     
    __shared__ uint32_t f_id,t_id,f_l,t_l,f_p,t_p,l_n,h_n;
    __shared__ T *a1,*a2,*a3;
    
    if(threadIdx.x==0)
    {
        f_id=fids[blockIdx.x];
        t_id=tids[blockIdx.x];
        f_l=ll[f_id];
        t_l=ll[t_id];
        f_p=pp[f_id];
        t_p=pp[t_id];
        l_n=(f_l<t_l)?f_l:t_l;
        h_n=(f_l>t_l)?f_l:t_l;
        a1=&base[0*l_n];//base+(0*l_n) //blockDim.x
        a2=&base[1*l_n];//base+(1*l_n) //blockDim.x
        a3=&base[2*l_n];//base+(2*l_n) //blockDim.x
                
        a1[0]=dist_sqr(0,0,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
        T d1=dist_sqr(0,1,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
        T d2=dist_sqr(1,0,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
        a2[0]=(a1[0]>d1)?a1[0]:d1;
        a2[1]=(a1[0]>d2)?a1[0]:d2;
if(CUDEBUG)
{
        printf("fid=%d t_id=%d f_l=%d t_l=%d f_p=%d t_p=%d\n",f_id,t_id,f_l,t_l,f_p,t_p);
        printf("after init: l_n=%d h_n=%d\n",l_n,h_n);
}               

     }        
     __syncthreads();

     for(uint32_t k=1;k<l_n-1;k++)
     {
         for(uint32_t m=0;m<k;m+=blockDim.x)
         {
             if(m+threadIdx.x<k)
             {             
 		const uint32_t s=(m+threadIdx.x)+1;
   		const uint32_t t=k-(m+threadIdx.x);
   		T t3=dist_sqr(s,t,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
                a3[m+threadIdx.x+1]=dmax(a1,a2,m+threadIdx.x,t3);
             }
             __syncthreads();
         }
         __syncthreads();
         
         if(threadIdx.x==0)
         {
            //update the two ends
            T d1=dist_sqr(0,k+1,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
            T d2=dist_sqr(k+1,0,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
            a3[0]=(a2[0]>d1)?a2[0]:d1;
            a3[k+1]=(a2[k]>d2)?a2[k]:d2;           
            T *temp=a1;
            a1=a2;
            a2=a3;
            a3=temp;
         }
         __syncthreads();
  
if(CUDEBUG)
{
     if(threadIdx.x==0)
     {     
        printf("s1:k=%d\n",k);
        for(int j=0;j<l_n;j++)
            printf("%10.5f",a1[j]);                
        printf("\n");
       for(int j=0;j<l_n;j++)
            printf("%10.5f",a2[j]);                
        printf("\n");         
     }
     __syncthreads();

}
     }    

     for(uint32_t k=l_n-1;k<h_n;k++)
     {
	 if(threadIdx.x==0)
	 {
            T d1=dist_sqr(0,k+1,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
            a3[0]=(a2[0]>d1)?a2[0]:d1;
	 }
	 __syncthreads();
	 
	 for(uint32_t m=0;m<k;m+=blockDim.x)
	 {
	     if(m+threadIdx.x<l_n-1)  
             {
   		const uint32_t s=(m+threadIdx.x)+1;
   		const uint32_t t=k-(m+threadIdx.x);
   		T t3=dist_sqr(s,t,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
                a3[m+threadIdx.x+(k!=h_n-1)]=dmax(a1,a2,m+threadIdx.x,t3);
              }
             __syncthreads();    
        }
        __syncthreads(); 
        
        if(threadIdx.x==0)
        {
            T *temp=a1;
            a1=a2;
            a2=a3;
            a3=temp;
        }
        __syncthreads();
if(CUDEBUG)
{
     if(threadIdx.x==0)
     {     
        printf("s2:k=%d\n",k);
        for(int j=0;j<l_n;j++)
            printf("%10.5f",a1[j]);                
        printf("\n");
        for(int j=0;j<l_n;j++)
            printf("%10.5f",a2[j]);                
        printf("\n");         
        } 
     __syncthreads();

}        
    }
    
    for(uint32_t k=h_n;k<h_n+l_n-2;k++)
    {
	 for(uint32_t m=0;m<k;m+=blockDim.x)
	 {
            if(m+threadIdx.x<h_n+l_n-2-k)
            {
                uint32_t s=k-h_n+2+(m+threadIdx.x);
                uint32_t t=h_n-1-(m+threadIdx.x);
                T t3=dist_sqr(s,t,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
                //m has a lower-left to bottom-up direction
                a3[m+threadIdx.x]=dmax(a1+1,a2,m+threadIdx.x,t3);
             }
             __syncthreads();
         }
         __syncthreads();
        
        if(threadIdx.x==0)
        {
            T *temp=a1;
            a1=a2;
            a2=a3;
            a3=temp;
        }
        __syncthreads();
if(CUDEBUG)
{
     if(threadIdx.x==0)
     {     
        printf("s3:k=%d\n",k);
        for(int j=0;j<l_n;j++)
            printf("%10.5f",a1[j]);                
        printf("\n");
        for(int j=0;j<l_n;j++)
            printf("%10.5f",a2[j]);                
        printf("\n");         
        } 
     __syncthreads();

}        
    }
    
    if(threadIdx.x==0)
        dis[blockIdx.x]= a2[0];    
    __syncthreads();
}
    