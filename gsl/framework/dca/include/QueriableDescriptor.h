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

#ifndef QUERIABLEDESCRIPTOR_H
#define QUERIABLEDESCRIPTOR_H
#include "Copyright.h"

#include <string>


class Queriable;

class QueriableDescriptor
{

   public:

      QueriableDescriptor();
      QueriableDescriptor(QueriableDescriptor&);
      QueriableDescriptor& operator=(const QueriableDescriptor& QD);
      std::string getName();
      void setName(std::string name);
      std::string getDescription();
      void setDescription(std::string description);
      std::string getType();
      void setType(std::string type);
      Queriable* getQueriable();
      void setQueriable(Queriable* ptrQueriable);
      ~QueriableDescriptor();

   private:

      std::string _name;
      std::string _description;
      std::string _type;
      Queriable* _ptrQueriable;
};
#endif
