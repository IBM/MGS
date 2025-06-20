// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_inAttrPSet_H
#define C_inAttrPSet_H
#include "Mdl.h"

#include "C_struct.h"
#include <memory>

class C_dataTypeList;
class C_generalList;

class C_inAttrPSet : public C_struct {
   protected:
      using C_struct::duplicate;
   public:
      virtual void addToList(C_generalList* gl);
      C_inAttrPSet(C_dataTypeList* dtl);
      virtual void duplicate(std::unique_ptr<C_inAttrPSet>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_struct>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_inAttrPSet();      
};


#endif // C_inAttrPSet_H
