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
      virtual void getGranuleMapper(std::vector<DataItem*> const & args, std::auto_ptr<GranuleMapper>&)=0;
      virtual void duplicate(std::auto_ptr<GranuleMapperType>& dup) const = 0;
      virtual void getInstance(std::auto_ptr<DataItem> &, 
			       std::vector<DataItem*> const *, 
			       LensContext* c = 0);
      virtual void getInstance(std::auto_ptr<DataItem> & adi, 
			       const NDPairList& ndplist,
			       LensContext* c);
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
