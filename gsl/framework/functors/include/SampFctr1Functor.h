// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SAMPFCTR1FUNCTOR_H
#define SAMPFCTR1FUNCTOR_H
#include "Copyright.h"

#include "Functor.h"
class SampFctr1Functor : public Functor
{
   public:
      virtual  const char * getCategory();
      static const char* _category;
      virtual ~SampFctr1Functor();
};
#endif
