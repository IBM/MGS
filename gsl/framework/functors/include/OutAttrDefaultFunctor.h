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

#ifndef _OUTATTRDEFAULTFUNCTOR_H_
#define _OUTATTRDEFAULTFUNCTOR_H_
#include "Copyright.h"

#include <memory>
#include <list>
#include <vector>
#include <string>
class DataItem;
class LensContext;
class ParameterSet;
#include "Functor.h"

class OutAttrDefaultFunctor: public Functor
{
   public:
      OutAttrDefaultFunctor();
      OutAttrDefaultFunctor(const OutAttrDefaultFunctor&);
      virtual void duplicate (std::unique_ptr<Functor>&& fap) const;
      virtual ~OutAttrDefaultFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      std::unique_ptr<ParameterSet> _pset;
      std::string _nodeModelName;
};
#endif
