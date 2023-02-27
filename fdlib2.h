#include <stdint.h>
double _st(double *a, const uint32_t m,const uint32_t i, const uint32_t j,
    const double *xx1,const double *yy1, const double *xx2,const double *yy2);

double _tr(double *a,const uint32_t fid, const uint32_t tid,
    const double *xx,const double *yy,const uint32_t *ll,const uint32_t *pp);

double _dp(double *a, const uint32_t m,const uint32_t i, const uint32_t j,
    const double *xx1,const double *yy1, const double *xx2,const double *yy2);

double _nt(const uint32_t fid, const uint32_t tid,
    const double *xx,const double *yy,const uint32_t *ll,const uint32_t *pp);
