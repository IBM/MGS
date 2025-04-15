// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GAUSSIANFUNCTORTYPE_H
#define GAUSSIANFUNCTORTYPE_H
#include "Copyright.h"

#include "FunctorType.h"

class GaussianFunctorType : public FunctorType
{
   public:
      GaussianFunctorType();
      void getFunctor(std::unique_ptr<Functor> & r_aptr);
      virtual std::string getName();
      virtual std::string getDescription();
      virtual void getQueriable(std::unique_ptr<InstanceFactoryQueriable>& dup);
      Functor* getFunctor();
      ~GaussianFunctorType();
};
#endif
