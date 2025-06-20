// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PRINTARGSFUNCTOR_H
#define PRINTARGSFUNCTOR_H
#include "Copyright.h"

#include "Functor.h"
#include <vector>
#include <memory>

class PrintFunctor: public Functor
{
   public:
      PrintFunctor();
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~PrintFunctor();
   protected:
      virtual void doInitialize(GslContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(GslContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
};
#endif
