// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_userFunction_H
#define C_userFunction_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_identifierList;

class C_userFunction : public C_general {
   protected:
      using C_general::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_userFunction();
      C_userFunction(C_identifierList* identifierList); 
      C_userFunction(const C_userFunction& rv);
      C_userFunction& operator=(const C_userFunction& rv);
      virtual void duplicate(std::unique_ptr<C_userFunction>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_userFunction();
      
   private:
      void copyOwnedHeap(const C_userFunction& rv);
      void destructOwnedHeap();
      C_identifierList* _identifierList;
};


#endif // C_userFunction_H
