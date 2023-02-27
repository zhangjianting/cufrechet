#include <stdio.h>
#include <time.h>
#include <sys/time.h>

template<class T>
size_t read_field(const char *filename,T*& field)
{
    FILE *fp{nullptr};
    if((fp=fopen(filename,"rb"))==nullptr)
    assert(fp!=nullptr);

    fseek (fp , 0 , SEEK_END);
    size_t sz=ftell (fp);
    assert(sz%sizeof(T)==0);
    size_t num_ringec = sz/sizeof(T);
    printf("num_rec=%zu",num_ringec);
    fseek (fp , 0 , SEEK_SET);

    field=new T[num_ringec];
    assert(field!=nullptr);
    size_t t=fread(field,sizeof(T),num_ringec,fp);
    assert(t==num_ringec);
    fclose(fp);
    return num_ringec;
}

float calc_time(const char *msg,timeval t0, timeval t1);
