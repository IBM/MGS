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

#ifndef Phase_H
#define Phase_H
#include "Copyright.h"

#include <memory>
#include <string>

class Simulation;

class Phase {

   public:
      Phase(const std::string& name);
      virtual void duplicate(std::auto_ptr<Phase>& rv) const = 0;
      virtual ~Phase();
     
      virtual std::string getType() const = 0;
      std::string getName() const {
	 return _name;
      }
      void setName(const std::string& name) {
	 _name = name;
      }
      virtual void addToSimulation(Simulation* sim) const =0;
      
   protected:
      std::string _name;
};


#endif // Phase_H
