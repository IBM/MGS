// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. and EPFL 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef ZEROTOUCHANALYSIS_H
#define ZEROTOUCHANALYSIS_H

#include <mpi.h>
#include "TouchAnalysis.h"
#include "Translation.h"
#include "Rotation.h"

class TouchTable;
class Tissue;
class Neuron;

class ZeroTouchAnalysis : public TouchAnalysis
{
	public:
		ZeroTouchAnalysis(Tissue* tissue);
		bool analyze(std::list<Translation>& translations, std::list<Rotation>& rotations);
		~ZeroTouchAnalysis();						

	private:
		Tissue* _tissue;
		Neuron* _neuronPtr;
		Neuron* _neuronEnd;
};

#endif
