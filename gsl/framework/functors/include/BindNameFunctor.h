// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _BINDNAMEFLOAT_FUNCTOR_H_
#define _BINDNAMEFLOAT_FUNCTOR_H_
#include "Copyright.h"

#include <string>
#include <vector>

#include "NDPairListFunctor.h"

class DataItem;

class BindNameFunctor: public NDPairListFunctor
{
   public:
      typedef std::pair<std::string, DataItem*> NDPairGenerator;
      BindNameFunctor();
      BindNameFunctor(const BindNameFunctor &);
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~BindNameFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      std::vector<NDPairGenerator> _nameDataItems;
};
#endif
