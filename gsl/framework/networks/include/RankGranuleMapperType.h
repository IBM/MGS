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

#ifndef RANKGRANULEMAPPERTYPE_H
#define RANKGRANULEMAPPERTYPE_H
#include "Copyright.h"

#include "GranuleMapperType.h"

#include <vector>

class Simulation;
class GranuleMapper;
class DataItem;
class NDPairList;

class RankGranuleMapperType : public GranuleMapperType
{
   public:
      RankGranuleMapperType(Simulation& s);
      void getGranuleMapper(std::vector<DataItem*> const & args, std::auto_ptr<GranuleMapper>&);
      virtual void duplicate(std::auto_ptr<GranuleMapperType>& dup) const;
      virtual ~RankGranuleMapperType();
      virtual void getQueriable(std::auto_ptr<InstanceFactoryQueriable>& dup);
};
#endif
