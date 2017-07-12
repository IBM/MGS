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

#ifndef C_general_H
#define C_general_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;

class C_general : public C_production {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_general();
      C_general(const C_general& rv);
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_general();

};


#endif // C_general_H
