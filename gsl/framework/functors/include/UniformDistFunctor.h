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

#ifndef _UNIFORMDISTFUNCTOR_H_
#define _UNIFORMDISTFUNCTOR_H_
#include "Copyright.h"

#include "Functor.h"

class UniformDistFunctor: public Functor
{
   public:
      UniformDistFunctor();
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~UniformDistFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
   private:
      float _minLim;
      float _maxLim;
};
#endif
