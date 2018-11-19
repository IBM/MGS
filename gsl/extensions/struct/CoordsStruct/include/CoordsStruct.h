// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-11-19-2018
//
//  (C) Copyright IBM Corp. 2005-2018  All rights reserved   .
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CoordsStruct_H
#define CoordsStruct_H

#include "Lens.h"
#include "DataItem.h"
#include "DataItemArrayDataItem.h"
#include "IntArrayDataItem.h"
#if defined(HAVE_MPI)
#include "OutputStream.h"
#endif
#include "ShallowArray.h"
#include "Struct.h"
#include "SyntaxErrorException.h"
#include "UnsignedIntDataItem.h"
#include <cassert>
#include <iostream>
#include <memory>

class CoordsStruct : public Struct
{
   public:
      CoordsStruct();
      virtual ~CoordsStruct();
      virtual void duplicate(std::unique_ptr<CoordsStruct>& dup) const;
      virtual void duplicate(std::unique_ptr<Struct>& dup) const;
      ShallowArray< unsigned, 3, 2 > coords;
   protected:
      virtual void doInitialize(LensContext *c, const std::vector<DataItem*>& args);
      virtual void doInitialize(const NDPairList& ndplist);
};

extern std::ostream& operator<<(std::ostream& os, const CoordsStruct& inp);
std::istream& operator>>(std::istream& is, CoordsStruct& inp);

#endif
