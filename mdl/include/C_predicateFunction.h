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

#ifndef C_predicateFunction_H
#define C_predicateFunction_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_identifierList;

class C_predicateFunction : public C_general {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_predicateFunction();
      C_predicateFunction(C_identifierList* identifierList); 
      C_predicateFunction(const C_predicateFunction& rv);
      C_predicateFunction& operator=(const C_predicateFunction& rv);
      virtual void duplicate(std::auto_ptr<C_predicateFunction>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_predicateFunction();
      
   private:
      void copyOwnedHeap(const C_predicateFunction& rv);
      void destructOwnedHeap();
      C_identifierList* _identifierList;
};


#endif // C_predicateFunction_H
