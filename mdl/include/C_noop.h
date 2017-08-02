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

#ifndef C_noop_H
#define C_noop_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>

class MdlContext;
class C_typeClassifier;
class C_generalList;

class C_noop : public C_general {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_noop();
      C_noop(const C_noop& rv);
      virtual void duplicate(std::auto_ptr<C_noop>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_noop();
      
};


#endif // C_noop_H
