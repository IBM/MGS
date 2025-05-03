// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _GAUSSIANFUNCTOR_H_
#define _GAUSSIANFUNCTOR_H_
#include "Copyright.h"

#include "Functor.h"

class GaussianFunctor: public Functor
{
   public:
      GaussianFunctor();
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~GaussianFunctor();
   protected:
      virtual void doInitialize(GslContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(GslContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      float _mean;
      float _stddev;
};
#endif
