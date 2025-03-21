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

#ifndef C_edge_H
#define C_edge_H
#include "Mdl.h"

#include "C_sharedCCBase.h"
#include <memory>

class MdlContext;

class C_edge : public C_sharedCCBase {
   using C_sharedCCBase::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_edge();
      C_edge(const std::string& name, C_interfacePointerList* ipl
			 , C_generalList* gl);
      C_edge(const C_edge& rv);
      virtual void duplicate(std::unique_ptr<C_edge>&& rv) const;
      virtual ~C_edge();
};


#endif // C_edge_H
