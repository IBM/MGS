// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef RANGEPASSTHRUFUNCTORTYPE_H
#define RANGEPASSTHRUFUNCTORTYPE_H
#include "Copyright.h"

#include "FunctorType.h"

class RangePassThruFunctorType : public FunctorType
{
   public:
      RangePassThruFunctorType();
      void getFunctor(std::unique_ptr<Functor> & r_aptr);
      virtual std::string getName();
      virtual std::string getDescription();
      virtual void getQueriable(std::unique_ptr<InstanceFactoryQueriable>& dup);
      Functor* getFunctor();
      ~RangePassThruFunctorType();
};
#endif
