// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SAMPFCTR2FUNCTOR_H
#define SAMPFCTR2FUNCTOR_H
#include "Copyright.h"

#include "Functor.h"
class SampFctr2Functor : public Functor
{
   public:
      virtual const std::string& getCategory() const;
      static const std::string _category;
};
#endif
