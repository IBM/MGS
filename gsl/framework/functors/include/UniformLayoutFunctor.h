// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~UniformLayoutFunctor();

   private:
      // methods
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);

      // members
      IntArrayDataItem _density;

};
#endif
