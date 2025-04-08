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
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      std::vector<float> *_float_array;
      std::vector<int> *_int_array;
      std::vector<DataItem*> *_di_array;
};
#endif
