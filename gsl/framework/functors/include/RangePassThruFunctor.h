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
