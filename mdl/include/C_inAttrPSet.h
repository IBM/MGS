// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_inAttrPSet_H
#define C_inAttrPSet_H
#include "Mdl.h"

#include "C_struct.h"
#include <memory>

class C_dataTypeList;
class C_generalList;

class C_inAttrPSet : public C_struct {

   public:
      virtual void addToList(C_generalList* gl);
      C_inAttrPSet(C_dataTypeList* dtl);
      virtual void duplicate(std::auto_ptr<C_inAttrPSet>& rv) const;
      virtual void duplicate(std::auto_ptr<C_struct>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_inAttrPSet();      
};


#endif // C_inAttrPSet_H
