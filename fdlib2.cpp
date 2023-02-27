#include <stdio.h>
#include <stdint.h>
#include <algorithm>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include "fdlib1.h"

#define DEBUG 0

using namespace std;

inline double dist_sqr(const uint32_t i,const uint32_t j,
    const double *xx1,const double *yy1,const double *xx2,const double *yy2)
{

	double fd=(xx1[i]-xx2[j])*(xx1[i]-xx2[j])+(yy1[i]-yy2[j])*(yy1[i]-yy2[j]);
if(DEBUG)
{
    printf("dist_sqr: s=%d t=%d,d=%10.5f\n",i,j,fd);
}
    return fd;
}

double _st(double *a, const uint32_t m,const uint32_t i, const uint32_t j,
    const double *xx1,const double *yy1, const double *xx2,const double *yy2)
{
    a[0*m+0]=dist_sqr(0,0,xx1,yy1,xx2,yy2);

    for(int k=1;k<=j;k++)
        a[0*m+k]=std::max(a[0*m+k-1],dist_sqr(0,k,xx1,yy1,xx2,yy2));
    for(int k=1;k<=i;k++)
        a[k*m+0]=std::max(a[(k-1)*m+0],dist_sqr(k,0,xx1,yy1,xx2,yy2));

    for(int s =1 ; s <=i; s++) {
            for(int t =1 ; t <=j; t++) {
                 double t1=std::min(a[(s-1)*m+t], a[(s-1)*m+(t-1)]);
                 double t2=std::min(t1, a[s*m+(t-1)]);
                 a[s*m+t] = std::max(t2,dist_sqr(s,t,xx1,yy1,xx2,yy2));
            }
        }
if(DEBUG)
{
    printf("_st\n");
    for(int s=0;s<=i;s++)
    {
        for(int t=0;t<=j;t++)
            printf("%10.5f",a[s*m+t]);
        printf("\n");
    }
    return(a[i*m+j]);
 }
}

double _tr(double *a,const uint32_t fid, const uint32_t tid,
    const double *xx,const double *yy,const uint32_t *ll,const uint32_t *pp)
{
//#undef DEBUG
//#define DEBUG fid==0&&tid==1

    int   f_l=ll[fid];
    int   t_l=ll[tid];
    int   f_p=pp[fid];
    int   t_p=pp[tid];
    int   l_n=(f_l<t_l)?f_l:t_l;
    int   h_n=(f_l>t_l)?f_l:t_l;

    //printf("_tr: f_l=%d t_l=%d\n",f_l,t_l);

    a[0]=dist_sqr(0,0,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
    //printf("_tr: a[0]=%15.10f\n",a[0]);
	//init first row
	for(uint32_t k=1;k<t_l;k++)
	{
		double d0= a[0*t_l+(k-1)];
		double d1= dist_sqr(0,k,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
		a[0*t_l+k]=(d0>d1)?d0:d1;
	}
	for(uint32_t k=1;k<f_l;k++)
	{
	    double d0= a[(k-1)*t_l+0];
	    double d1= dist_sqr(k,0,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
	    a[k*t_l+0]=(d0>d1)?d0:d1;
    }
if(DEBUG)
{
    for(int i=0;i<f_l;i++)
    {
        for(int j=0;j<t_l;j++)
            printf("%10.5f",a[i*t_l+j]);
        printf("\n");
    }
}
    //printf("_tr: begin\n");
    for(uint32_t k=1;k<l_n-1;k++)
    {
        for(uint32_t m=0;m<k;m++) //for each of the 0...k-1 CUDA threads
        {
            uint32_t s=m+1; //row
            uint32_t t=k-m; //col
            double t1=(a[(s-1)*t_l+t]< a[(s-1)*t_l+(t-1)])?a[(s-1)*t_l+t]:a[(s-1)*t_l+(t-1)];
            double t2=(t1< a[s*t_l+(t-1)])?t1:a[s*t_l+(t-1)];
            double t3=dist_sqr(s,t,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
            a[s*t_l+t]=(t2>t3)?t2:t3;
 //if(DEBUG)
 //           printf("k=%d m=%d s=%d t=%d t1=%10.5f t2=%10.5f t3=%10.5f r=%10.5f\n",k,m,s,t,t1,t2,t3,a[s*t_l+t]);
         }
    }

if(DEBUG)
{
    printf("after 1st seg\n");
    for(int i=0;i<f_l;i++)
    {
        for(int j=0;j<t_l;j++)
            printf("%10.5f",a[i*t_l+j]);
        printf("\n");
    }
}

    for(uint32_t k=l_n-1;k<h_n;k++)
    {
	   for(uint32_t m=0;m<l_n-1;m++) //for each of the 0...(h_n-l_n+1) CUDA threads
       {
            uint32_t s=(f_l<=t_l)?(m+1):(k-m);//row
            uint32_t t=(f_l<=t_l)?(k-m):(m+1);//col
            double t1=(a[(s-1)*t_l+t]< a[(s-1)*t_l+(t-1)])?a[(s-1)*t_l+t]:a[(s-1)*t_l+(t-1)];
            double t2=(t1< a[s*t_l+(t-1)])?t1:a[s*t_l+(t-1)];
            double t3=dist_sqr(s,t,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
            a[s*t_l+t]=(t2>t3)?t2:t3;
            //printf("k=%5d m=%5d s=%5d t=%5d d=%10.5f\n",k,m,s,t,a[s*t_l+t]);
        }
    }
if(DEBUG)
{
    printf("after 2nd seg\n");
    for(int i=0;i<f_l;i++)
    {
        for(int j=0;j<t_l;j++)
            printf("%10.5f",a[i*t_l+j]);
        printf("\n");
    }
}

    for(uint32_t k=l_n-2;k>0;k--)
    {
        for(uint32_t m=0;m<k;m++) //for each of the 0...k-1 CUDA threads
        {
            uint32_t s=(f_l<=t_l)?l_n-1-m:h_n-k+m;
            uint32_t t=(f_l<=t_l)?h_n-k+m:l_n-1-m;
            double t1=(a[(s-1)*t_l+t]< a[(s-1)*t_l+(t-1)])?a[(s-1)*t_l+t]:a[(s-1)*t_l+(t-1)];
            double t2=(t1< a[s*t_l+(t-1)])?t1:a[s*t_l+(t-1)];
            double t3=dist_sqr(s,t,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
            a[s*t_l+t]=(t2>t3)?t2:t3;
            //printf("k=%5d m=%5d s=%5d t=%5d d=%10.5f\n",k,m,s,t,a[s*t_l+t]);
         }
	 }
if(DEBUG)
{
    printf("after 3rd seg\n");
    for(int i=0;i<f_l;i++)
    {
        for(int j=0;j<t_l;j++)
            printf("%10.5f",a[i*t_l+j]);
        printf("\n");
    }
}

	 //printf("_tr: res=%15.10f\n",a[f_l*t_l-1]);
	 return(a[f_l*t_l-1]);
}

inline double dmax(const double *a1,const double *a2,const uint32_t m,const double t3 )
{
    //double d1=b_work[(s-1)*t_l+t];
    //double d2=b_work[(s-1)*t_l+(t-1)];
    //double d3=b_work[s*t_l+(t-1)];
    double d1=a2[m]; //(s-1,t)
    double d2=a1[m]; //(s-1,t-1)
    double d3=a2[m+1];//(s,t-1)

	double t1=(d1<d2)?d1:d2;
	double t2=(t1<d3)?t1:d3;
    double d=(t3>t2)?t3:t2;
if(DEBUG)
{
    printf("dmax: m=%d d1=%10.5f d2=%10.5f d3=%10.5f t3=%10.5f d=%10.5f\n",m,d1,d2,d3,t3,d);
}
    return(d);

}

double _nt(const uint32_t fid, const uint32_t tid,
    const double *xx,const double *yy,const uint32_t *ll,const uint32_t *pp)
{
//#undef DEBUG
//#define DEBUG fid==0&&tid==1

    int   f_l=ll[fid];
    int   t_l=ll[tid];
    assert(f_l<=t_l);

    int   f_p=pp[fid];
    int   t_p=pp[tid];
    int   l_n=(f_l<t_l)?f_l:t_l;
    int   h_n=(f_l>t_l)?f_l:t_l;
    double *a1=new double[l_n];
    double *a2=new double[l_n];
    double *a3=new double[l_n];
    assert(a1!=NULL && a2!=NULL && a3!=NULL);

    a1[0]=dist_sqr(0,0,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
    double d1=dist_sqr(0,1,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
    double d2=dist_sqr(1,0,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
    a2[0]=(a1[0]>d1)?a1[0]:d1;
    a2[1]=(a1[0]>d2)?a1[0]:d2;
 if(DEBUG)
{
   printf("_tr: f_l=%d t_l=%d\n",f_l,t_l);
   printf("a1[0]=%10.5f a2[0]=%10.5f a2[1]=%10.5f\n",a1[0],a2[0],a2[1]);
}
    //lower-left triangle
    for(uint32_t k=1;k<l_n-1;k++)
    {
        for(uint32_t m=0;m<k;m++)
       {
		   	const uint32_t s=m+1;
		   	const uint32_t t=k-m;
		   	double t3=dist_sqr(s,t,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
            a3[m+1]=dmax(a1,a2,m,t3);
		}
        //update the two ends
        d1=dist_sqr(0,k+1,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
        d2=dist_sqr(k+1,0,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
        a3[0]=(a2[0]>d1)?a2[0]:d1;
        a3[k+1]=(a2[k]>d2)?a2[k]:d2;
        double *tmp=a1;
        a1=a2;
        a2=a3;
        a3=tmp;
        memset(a3,0,sizeof(double)*(l_n));

if(DEBUG)
{
        for(uint32_t m=0;m<k+2;m++)
            printf("%10.5f ",a2[m]);
        printf("\n");
}
     }

    for(uint32_t k=l_n-1;k<h_n;k++)
    {
        if(k!=h_n-1)
        {
            d1=dist_sqr(0,k+1,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
            a3[0]=(a2[0]>d1)?a2[0]:d1;
	    }
        for(uint32_t m=0;m<l_n-1;m++)
        {
  			double t3=dist_sqr(m+1,k-m,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
            a3[m+(k!=h_n-1)]=dmax(a1,a2,m,t3);
		}

       double *tmp=a1;
       a1=a2;
       a2=a3;
       a3=tmp;
       memset(a3,0,sizeof(double)*(l_n));
if(DEBUG)
{
        for(uint32_t m=0;m<l_n;m++)
            printf("%10.5f ",a2[m]);
        printf("\n");
}
    }

    for(uint32_t k=h_n;k<h_n+l_n-2;k++)
    {
        for(uint32_t m=0;m<h_n+l_n-2-k;m++)
        {
            uint32_t s=k-h_n+2+m;
            uint32_t t=h_n-1-m;
            double t3=dist_sqr(s,t,xx+f_p,yy+f_p,xx+t_p,yy+t_p);
            //m has a lower-left to bottom-up direction
             a3[m]=dmax(a1+1,a2,m,t3);
	    }
        double *tmp=a1;
        a1=a2;
        a2=a3;
        a3=tmp;
        memset(a3,0,sizeof(double)*(l_n));

if(DEBUG)
{
        for(uint32_t m=0;m<h_n+l_n-2-k;m++)
            printf("%10.5f ",a2[m]);
        printf("\n");
}
 	}
     double d=a2[0];
	 delete[] a1;
	 delete[] a2;
	 delete[] a3;
	 return(d);
}


double _dp(double *a, const uint32_t m,const uint32_t i, const uint32_t j,
    const double *xx1,const double *yy1, const double *xx2,const double *yy2)
{
    //printf("_dp: m=%d i=%d j=%d\n",m,i,j);
    //assert(i>=0 && i<m && j>=0 && j<m);
    double fd=0;
    if (a[i*m+j] > -1) fd= a[i*m+j];
    else if (i == 0 && j == 0) fd=dist_sqr(0,0,xx1,yy1,xx2,yy2);
    else if (i > 0 && j == 0) fd= std::max(_dp(a,m, i-1, 0, xx1,yy1,xx2,yy2), dist_sqr(i,0,xx1,yy1,xx2,yy2));
    else if (i == 0 && j > 0) fd= std::max(_dp(a,m, 0, j-1, xx1,yy1,xx2,yy2), dist_sqr(0,j,xx1,yy1,xx2,yy2));
    else {
         assert(i>0 && j>0);
         double t1=std::min(_dp(a,m, i-1, j, xx1,yy1,xx2,yy2), _dp(a,m, i-1, j-1, xx1,yy1,xx2,yy2));
	     double t2=std::min(t1,_dp(a, m,i, j-1, xx1,yy1,xx2,yy2));
	     fd = std::max(t2,dist_sqr(i,j,xx1,yy1,xx2,yy2));
	}
	//printf("_dp: i=%d j=%d,fd=%20.15f\n",i,j,fd);
    a[i*m+j]=fd;
    return fd;
}
