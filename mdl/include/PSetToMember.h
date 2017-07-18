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

#ifndef PSetToMember_H
#define PSetToMember_H
#include "Mdl.h"

#include <memory>
#include <vector>
#include <set>
#include "DataType.h"

class StructType;

class PSetToMember {

   public:
      typedef std::pair<std::string, DataType*> elemType;
      enum MappingType {ONETOONE, ONETOMANY};

      typedef std::vector<elemType>::const_iterator 
      const_iterator;
      typedef std::vector<elemType>::iterator iterator;

      PSetToMember(StructType* pset = 0);

      PSetToMember(const PSetToMember& rv);
      PSetToMember& operator=(const PSetToMember& rv);

      virtual void duplicate(std::auto_ptr<PSetToMember>& rv) const;
      virtual ~PSetToMember();
      void setPSet(StructType* pset) {
	 _pset = pset;
      }
      StructType* getPSet() {
	 return _pset;
      }
      void addMapping(const std::string& name, std::auto_ptr<DataType>& data);

      std::vector<elemType>& getMappings() {
	 return _mappings;
      }

      inline const_iterator begin() const {
	 return _mappings.begin();
      }
      inline const_iterator end() const {
	 return _mappings.end();
      }
      inline iterator begin() {
	 return _mappings.begin();
      }
      inline iterator end() {
	 return _mappings.end();
      }
      iterator find(const std::string& token);
      std::string getPSetToMemberString() const;
      std::string getPSetToMemberCode(
	 const std::string& tab, 
	 std::set<std::string>& requiredIncreases) const;
   private:
      void destructOwnedHeap();
      void copyOwnedHeap(const PSetToMember& rv);

      virtual void checkAndExtraWork(const std::string& name,
	 DataType* member, const DataType* pset);

      bool existsInMappings(const std::string& token) const;

      StructType* _pset;
      std::vector<elemType> _mappings;
      std::vector<MappingType> _mappingType;
};


#endif // PSetToMember_H
