// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _SUMFUNCTOR_H_
#define _SUMFUNCTOR_H_
#include "Copyright.h"

#include <vector>
#include "Functor.h"

class SumFunctor: public Functor
{
   public:
      SumFunctor();
      SumFunctor(const SumFunctor&);
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~SumFunctor();
   protected:
      virtual void doInitialize(GslContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(GslContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      std::vector<float> *_float_array;
      std::vector<int> *_int_array;
      std::vector<DataItem*> *_di_array;
};
#endif
