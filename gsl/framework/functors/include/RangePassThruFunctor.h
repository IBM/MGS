// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _RANGE_PASS_THRU_FUNCTOR_H_
#define _RANGE_PASS_THRU_FUNCTOR_H_
#include "Copyright.h"

#include <string>
#include <vector>
#include <memory>
#include "Functor.h"

class RangePassThruFunctor: public Functor
{
   public:
      RangePassThruFunctor();
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~RangePassThruFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      float  _left_limit;
      float  _right_limit;
      Functor *_testFunct;
      std::string _left_oper;
      std::string _right_oper;
};
#endif
