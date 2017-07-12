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

#ifndef QUERYFIELD_H
#define QUERYFIELD_H
#include "Copyright.h"

#include <string>
#include <vector>
#include <memory>


class EnumEntry;

class QueryField
{

   public:
      enum queryType{VALUE, ENUM};
      QueryField(queryType);
      QueryField(QueryField*);
      std::string getField();
      void setField(std::string field);
      std::string getName();
      void setName(std::string name);
      std::string getDescription();
      void setDescription(std::string description);
      queryType getType();
      std::vector<EnumEntry*> const & getEnumEntries() const;
      void addEnumEntry(std::auto_ptr<EnumEntry>&);
      std::string getFormat();
      void setFormat(std::string format);
      bool isSet();
      ~QueryField();

   private:
      queryType _type;
      std::string _field;
      std::string _name;
      std::string _description;
      std::vector<EnumEntry*> _enumEntries;
      std::string _format;
};
#endif
