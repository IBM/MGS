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

#ifndef C_phaseIdentifier_H
#define C_phaseIdentifier_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <string>
#include <cassert>

class MdlContext;
class C_identifierList;

class C_phaseIdentifier : public C_production {

   public:
      virtual void execute(MdlContext* context);
      C_phaseIdentifier(); 
      C_phaseIdentifier(const std::string& name); 
      C_phaseIdentifier(const std::string& name,
			C_identifierList* identifierList); 
      C_phaseIdentifier(const C_phaseIdentifier& rv);
      C_phaseIdentifier& operator=(const C_phaseIdentifier& rv);
      virtual void duplicate(std::unique_ptr<C_phaseIdentifier>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_production>&& rv) const;
      virtual ~C_phaseIdentifier();

      const std::string& getName() const {
	 return _name;
      }

      const std::vector<std::string>& getIdentifiers() const;

   private:
      void copyOwnedHeap(const C_phaseIdentifier& rv);
      void destructOwnedHeap();

      std::string _name;
      C_identifierList* _identifierList;

      static const std::vector<std::string> _emptyIds;
};


#endif // C_phaseIdentifier_H
