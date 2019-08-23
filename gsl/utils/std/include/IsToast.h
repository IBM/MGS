// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef ISTOAST_H
#define ISTOAST_H

#include <sstream>
#include <cassert>
#include <iostream>
#include <math.h>

#ifndef TOAST_OFF
#define ISTOAST(x) if(isToast(x, (getSimulation().getIteration()))){assert(0);}
#else
#define ISTOAST(x)
#endif

template<class T> bool isToast(T f, int i)
{
  bool rval=false;
  if (std::isnan(f)) {
    std::cerr<<"NaN detected at mark "<<i<<"!!"<<std::endl;
    rval=true;
  }
  if (std::isinf(f)) {
    std::cerr<<"INF detected at mark "<<i<<"!!"<<std::endl;
    rval=true;
  }
  return rval;
}

template<class T> void isToast(T f)
{
  if (std::isnan(f)) {
    std::cerr<<"NaN detected!!"<<std::endl;
    assert(0);
  }
  if (std::isinf(f)) {
    std::cerr<<"INF detected!!"<<std::endl;
    assert(0);
  }
}

#endif

