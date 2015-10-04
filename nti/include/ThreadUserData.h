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
// ================================================================

#ifndef THREADUSERDATA
#define THREADUSERDATA

#include <mpi.h>
#include "TouchVector.h"
#include "Params.h"
#include "RNG.h"
#include "Decomposition.h"
#include "TouchSpace.h"
#include "NeurogenParams.h"

class ThreadUserData
{
   public:
        ThreadUserData(int nThreads) : _file(0), 
	  _parms(0), _E(0), _touchVectors(0), _nThreads(nThreads)
	{
	  _parms = new Params*[_nThreads];
	  _E = new double[_nThreads];
	  _touchVectors = new TouchVector[_nThreads];
	  _rangens = new RNG[_nThreads];
	  _decompositions = new Decomposition*[_nThreads];
	  _touchSpaces = new TouchSpace*[_nThreads];
	  for (int i=0; i<_nThreads; ++i) {
	    _E[i]=0;
	    _decompositions[i]=0;
	    _touchSpaces[i]=0;
	  }
	}
	
	void resetDecompositions(Decomposition* decomposition) {
	  for (int i=0; i<_nThreads; ++i) {
	    delete _decompositions[i];
	    _decompositions[i] = decomposition->duplicate();    
	  }
	}
	void resetEnergy() {
	  for (int i=0; i<_nThreads; ++i) {
	    _E[i]=0;
	  }
	}
	Params** _parms;
	FILE* _file;
	double* _E;
	TouchVector* _touchVectors;
	RNG* _rangens;
	Decomposition** _decompositions;
	TouchSpace** _touchSpaces;
	int _nThreads;

	~ThreadUserData() {
	  for (int i=0 ; i<_nThreads; ++i) delete _parms[i];
	  delete [] _parms;
	  delete [] _E;
	  delete [] _touchVectors;
	  delete [] _rangens;
	  for (int i=0 ; i<_nThreads; ++i) {
	    delete _decompositions[i];
	    delete _touchSpaces[i];
	  }
	  delete [] _decompositions;
	  delete [] _touchSpaces;
	}    
};
#endif
