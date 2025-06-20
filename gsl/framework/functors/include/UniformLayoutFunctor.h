// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _UNIFORMLAYOUTFUNCTOR_H_
#define _UNIFORMLAYOUTFUNCTOR_H_
#include "Copyright.h"
#include "LayoutFunctor.h"
#include "IntArrayDataItem.h"
#include <vector>

class UniformLayoutFunctor: public LayoutFunctor
{
   public:
      UniformLayoutFunctor();
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~UniformLayoutFunctor();

   private:
      // methods
      virtual void doInitialize(GslContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(GslContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);

      // members
      IntArrayDataItem _density;

};
#endif
