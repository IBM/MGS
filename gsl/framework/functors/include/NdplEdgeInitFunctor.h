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

#ifndef _NDPLEDGEINITFUNCTOR_H_
#define _NDPLEDGEINITFUNCTOR_H_
#include "Copyright.h"

#include "EdgeInitializerFunctor.h"
#include <memory>
#include <list>
#include <vector>
class DataItem;
class LensContext;
class Functor;
class ParameterSet;

class NdplEdgeInitFunctor: public EdgeInitializerFunctor
{
   public:
      NdplEdgeInitFunctor();
      NdplEdgeInitFunctor(const NdplEdgeInitFunctor&);
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~NdplEdgeInitFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
   private:
      std::auto_ptr<Functor> _functor_ap;
      std::auto_ptr<ParameterSet> _pset_ap;
};
#endif
