// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_psetMapping_H
#define C_psetMapping_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_identifierList;

class C_psetMapping : public C_general {

   public:
      using C_general::duplicate;
      virtual void execute(MdlContext* context);
      C_psetMapping();
      C_psetMapping(const std::string& psetMember,
		    C_identifierList* member); 
      virtual void duplicate(std::unique_ptr<C_psetMapping>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_psetMapping();

      C_psetMapping(const C_psetMapping& rv);
      C_psetMapping& operator=(const C_psetMapping& rv);

      const std::string& getPSetMember() const {
	 return _psetMember;
      }

      const std::string& getMember() const;
      
      bool getSubAttributePathExists() const;

      std::vector<std::string> getSubAttributePath() const;

   protected:
      void destructOwnedHeap();
      void copyOwnedHeap(const C_psetMapping& rv);

      std::string _psetMember;
      C_identifierList* _member;
};


#endif // C_psetMapping_H
