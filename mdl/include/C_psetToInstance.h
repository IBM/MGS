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

#ifndef C_psetToInstance_H
#define C_psetToInstance_H
#include "Mdl.h"

#include "C_psetMapping.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;

class C_psetToInstance : public C_psetMapping {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_psetToInstance();
      C_psetToInstance(const std::string& psetMember,
		       C_identifierList* member); 
      virtual void duplicate(std::auto_ptr<C_psetToInstance>& rv) const;
      virtual void duplicate(std::auto_ptr<C_psetMapping>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_psetToInstance();
};


#endif // C_psetToInstance_H
