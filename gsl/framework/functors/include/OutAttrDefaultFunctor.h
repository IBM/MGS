// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
