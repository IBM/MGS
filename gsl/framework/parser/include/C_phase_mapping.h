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

#ifndef C_phase_mapping_H
#define C_phase_mapping_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_phase_mapping : public C_production
{
   public:
      C_phase_mapping(const C_phase_mapping&);
      C_phase_mapping(const std::string& , const std::string& , 
		      SyntaxError *);
      virtual ~C_phase_mapping();
      virtual C_phase_mapping* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::string& getModelPhase() {
	 return _modelPhase;
      }
      const std::string& getSimulationPhase() {
	 return _simulationPhase;
      }

   private:
      std::string _modelPhase;
      std::string _simulationPhase;
};
#endif
