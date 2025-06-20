// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_identifierList_H
#define C_identifierList_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <vector>
#include <string>

class MdlContext;

class C_identifierList : public C_production {
   using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_identifierList();
      C_identifierList(const std::string& id);
      C_identifierList(C_identifierList* ids, const std::string& id);
      C_identifierList(const C_identifierList& rv);
      virtual void duplicate(std::unique_ptr<C_identifierList>&& rv) const;
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
