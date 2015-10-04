// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef TOUCHANALYSIS_H
#define TOUCHANALYSIS_H

#include <mpi.h>
#include "Translation.h"
#include "Rotation.h"

#include <vector>
#include <list>

class TouchTable;

class TouchAnalysis
{
	public:
		TouchAnalysis() {}
		std::vector<TouchTable*> & getTouchTables() {return _touchTables;}
		bool virtual analyze(std::list<Translation>& translations, std::list<Rotation>& rotations)=0;
		virtual ~TouchAnalysis() {}
		
	protected:
		std::vector<TouchTable*> _touchTables;
};

#endif
