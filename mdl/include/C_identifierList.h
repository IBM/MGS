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

#ifndef C_identifierList_H
#define C_identifierList_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <vector>
#include <string>

class MdlContext;

class C_identifierList : public C_production {

   public:
      virtual void execute(MdlContext* context);
      C_identifierList();
      C_identifierList(const std::string& id);
      C_identifierList(C_identifierList* ids, const std::string& id);
      C_identifierList(const C_identifierList& rv);
      virtual void duplicate(std::auto_ptr<C_identifierList>& rv) const;
      virtual ~C_identifierList();
      const std::vector<std::string>& getIdentifiers() {
	 return _identifiers;
      }

   private:
      std::string _identifier;
      C_identifierList* _identifierList;
      std::vector<std::string> _identifiers;
};


#endif // C_identifierList_H
