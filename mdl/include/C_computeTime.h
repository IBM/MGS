// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_computeTime_H
#define C_computeTime_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_identifierList;

class C_computeTime: public C_general {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_computeTime();
      C_computeTime(C_identifierList* identifierList); 
      C_computeTime(const C_computeTime& rv);
      C_computeTime(double& rv);
      C_computeTime& operator=(const C_computeTime& rv);
      virtual void duplicate(std::unique_ptr<C_computeTime>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_computeTime();
      
   private:
      void copyOwnedHeap(const C_computeTime& rv);
      void destructOwnedHeap();
      C_identifierList* _identifierList;
      double _computeTime;
};


#endif // C_computeTime_H
