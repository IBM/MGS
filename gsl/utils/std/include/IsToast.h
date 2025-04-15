// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

