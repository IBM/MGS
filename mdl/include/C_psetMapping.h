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
      virtual void execute(MdlContext* context);
      C_psetMapping();
      C_psetMapping(const std::string& psetMember,
		    C_identifierList* member); 
      virtual void duplicate(std::auto_ptr<C_psetMapping>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
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
