#ifndef DEVICESINFO_H
#define DEVICESINFO_H

typedef struct DevicesInfo
{
    int nCount;
    char** sNames;
} DevicesInfo;

#define MAX_GPUS 10

#endif // DEVICESINFO_H
