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

#ifndef C_edge_H
#define C_edge_H
#include "Mdl.h"

#include "C_sharedCCBase.h"
#include <memory>

class MdlContext;

class C_edge : public C_sharedCCBase {

   public:
      virtual void execute(MdlContext* context);
      C_edge();
      C_edge(const std::string& name, C_interfacePointerList* ipl
			 , C_generalList* gl);
      C_edge(const C_edge& rv);
      virtual void duplicate(std::auto_ptr<C_edge>& rv) const;
      virtual ~C_edge();
};


#endif // C_edge_H
