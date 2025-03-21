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
