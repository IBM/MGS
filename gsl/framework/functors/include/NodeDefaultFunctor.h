// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _NODEDEFAULTFUNCTOR_H_
#define _NODEDEFAULTFUNCTOR_H_
#include "Copyright.h"

#include "NodeInitializerFunctor.h"
#include <memory>
#include <list>
#include <vector>
class DataItem;
class GslContext;

class NodeDefaultFunctor: public NodeInitializerFunctor
{
   public:
      NodeDefaultFunctor();
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~NodeDefaultFunctor();
   protected:
      virtual void doInitialize(GslContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(GslContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
};
#endif
