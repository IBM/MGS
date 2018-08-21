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

#ifndef _SAMEFUNCTOR_H_
#define _SAMEFUNCTOR_H_
#include "Copyright.h"

#include "NodeInitializerFunctor.h"
#include <memory>
#include <list>
#include <vector>
class DataItem;
class LensContext;
class ParameterSet;

class SameFunctor: public NodeInitializerFunctor
{
   public:
      SameFunctor();
      SameFunctor(const SameFunctor&);
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~SameFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
   private:
      std::auto_ptr<ParameterSet> _pset;

};
#endif
