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
      virtual void duplicate(std::auto_ptr<C_outAttrPSet>& rv) const;
      virtual void duplicate(std::auto_ptr<C_struct>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_outAttrPSet();      
};


#endif // C_outAttrPSet_H
