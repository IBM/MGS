// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GRANULEMAPPERTYPE_H
#define GRANULEMAPPERTYPE_H
#include "Copyright.h"

//#include "Publishable.h"
#include "InstanceFactory.h"
//#include "NDPairList.h"
#include "GranuleMapper.h"
#include "DuplicatePointerArray.h"

#include <vector>
#include <memory>
#include <string>

class ParameterSet;
//class GranuleMapper;
class Publisher;
class NDPairList;
class Simulation;

class GranuleMapperType : public InstanceFactory
{
   public:
      GranuleMapperType(Simulation& s, const std::string& name = "", const std::string& description = "");
      virtual void getGranuleMapper(std::vector<DataItem*> const & args, std::unique_ptr<GranuleMapper>&)=0;
      virtual void duplicate(std::unique_ptr<GranuleMapperType>& dup) const = 0;
      virtual void getInstance(std::unique_ptr<DataItem> &, 
			       std::vector<DataItem*> const *, 
			       GslContext* c = 0);
      virtual void getInstance(std::unique_ptr<DataItem> & adi, 
			       const NDPairList& ndplist,
			       GslContext* c);
      virtual std::string getName() {return _name;}
      virtual std::string getDescription() {return _description;}
      virtual ~GranuleMapperType();

   protected:
      DuplicatePointerArray<GranuleMapper, 50> _granuleMapperList;
      std::string _name;
      std::string _description;
      Simulation& _sim;
};
#endif
