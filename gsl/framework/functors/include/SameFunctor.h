// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _SAMEFUNCTOR_H_
#define _SAMEFUNCTOR_H_
#include "Copyright.h"

#include "NodeInitializerFunctor.h"
#include <memory>
#include <list>
#include <vector>
class DataItem;
class GslContext;
class ParameterSet;

class SameFunctor: public NodeInitializerFunctor
{
   public:
      SameFunctor();
      SameFunctor(const SameFunctor&);
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~SameFunctor();
   protected:
      virtual void doInitialize(GslContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(GslContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      std::unique_ptr<ParameterSet> _pset;

};
#endif
