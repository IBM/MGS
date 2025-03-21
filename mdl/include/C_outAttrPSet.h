// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_outAttrPSet_H
#define C_outAttrPSet_H
#include "Mdl.h"

#include "C_struct.h"
#include <memory>

class C_dataTypeList;
class C_generalList;

class C_outAttrPSet : public C_struct {

   public:
      virtual void addToList(C_generalList* gl);
      C_outAttrPSet(C_dataTypeList* dtl);
      virtual void duplicate(std::unique_ptr<C_outAttrPSet>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_struct>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_outAttrPSet();      
};


#endif // C_outAttrPSet_H
