// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef _EDGEDEFAULTFUNCTOR_H_
#define _EDGEDEFAULTFUNCTOR_H_
#include "Copyright.h"

#include "EdgeInitializerFunctor.h"
#include <memory>
#include <list>
#include <vector>
class DataItem;
class LensContext;
class ParameterSet;

class EdgeDefaultFunctor: public EdgeInitializerFunctor
{
   public:
      EdgeDefaultFunctor();
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~EdgeDefaultFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
   private:
};
#endif
