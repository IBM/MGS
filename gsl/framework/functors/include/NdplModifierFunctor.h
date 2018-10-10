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

#ifndef _NDPLMODIFIERFUNCTOR_H_
#define _NDPLMODIFIERFUNCTOR_H_
#include "Copyright.h"

#include "Functor.h"
#include <memory>
#include <list>
#include <vector>
class DataItem;
class LensContext;
class Functor;
class ParameterSet;
class NDPairList;

class NdplModifierFunctor: public Functor
{
   public:
      NdplModifierFunctor();
      NdplModifierFunctor(const NdplModifierFunctor&);
      virtual void duplicate(std::unique_ptr<Functor> &fap) const;
      virtual ~NdplModifierFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      std::unique_ptr<Functor> _functor_ap;
      NDPairList *_ndpList;
};
#endif
