// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _TRAVERSEFUNCTOR_H_
#define _TRAVERSEFUNCTOR_H_
#include "Copyright.h"

#include "ConnectorFunctor.h"
#include <memory>
#include <list>
#include <vector>
class DataItem;
class LensContext;

class TraverseFunctor: public ConnectorFunctor
{
   public:
      TraverseFunctor();
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~TraverseFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
};
#endif
