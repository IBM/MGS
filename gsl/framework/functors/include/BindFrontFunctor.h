// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _BIND_FRONT_FUNCTOR_H_
#define _BIND_FRONT_FUNCTOR_H_
#include "Copyright.h"

#include <string>
#include <vector>
#include <memory>
#include "Functor.h"

class BindFrontFunctor: public Functor
{
   public:
      BindFrontFunctor();
      BindFrontFunctor(const BindFrontFunctor &);
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~BindFrontFunctor();
   protected:
      virtual void doInitialize(GslContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(GslContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      Functor *_bind_functor;
      std::vector<DataItem*> _bind_args;
};
#endif
