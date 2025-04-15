// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
