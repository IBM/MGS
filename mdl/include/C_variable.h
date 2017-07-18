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

#ifndef C_variable_H
#define C_variable_H
#include "Mdl.h"

#include "C_connectionCCBase.h"
#include <memory>

class MdlContext;

class C_variable : public C_connectionCCBase {

   public:
      virtual void execute(MdlContext* context);
      C_variable();
      C_variable(const std::string& name, C_interfacePointerList* ipl
			 , C_generalList* gl);
      C_variable(const C_variable& rv);
      virtual void duplicate(std::auto_ptr<C_variable>& rv) const;
      virtual ~C_variable();
};


#endif // C_variable_H
