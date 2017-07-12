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
// ================================================================

#ifndef RUNTIMETOPOLOGY_H
#define RUNTIMETOPOLOGY_H

#ifdef USING_BLUEGENEL
#include <rts.h>
#endif
#ifdef USING_BLUEGENEP
#include <spi/kernel_interface.h>
#include <common/bgp_personality.h>
#include <common/bgp_personality_inlines.h>
#endif
#ifdef USING_BLUEGENEQ
#include <spi/include/kernel/location.h>
#endif

/* * * Use any existing Topology.h * * */
#define READ_TOPOLOGY(a,b) (!a || b)
/* * * Overwrite any existing Topology.h * * */
//#define READ_TOPOLOGY(a,b) (!a && b)

#include <sstream>
#include <fstream>
#include <iterator>
#include <vector>
#include <stdlib.h>
#include <cassert>

//#include <spi/include/kernel/location.h>

class RunTimeTopology {
 public:
  RunTimeTopology()
    : _X(-1), _Y(-1), _Z(-1), _runTimeTopologyExists(true), _storedTopologyExists(false)
    {
#ifdef USING_BLUEGENEL
      BGLPersonality personality;
      rts_get_personality(&personality, sizeof(personality));
      _X=personality.xSize;
      _Y=personality.ySize;
      _Z=personality.zSize;
#else
#ifdef USING_BLUEGENEP
      _BGP_Personality_t personality;
      Kernel_GetPersonality(&personality, sizeof(personality));
      _X=personality.Network_Config.Xnodes;
      _Y=personality.Network_Config.Ynodes;
      _Z=personality.Network_Config.Znodes;  
#else
#ifdef USING_BLUEGENEQ
      //Personality_t personality;
      //Kernel_GetPersonality(&personality, sizeof(personality));
      BG_JobCoords_t block;
      int subblock, blocksize;
      
      subblock = Kernel_JobCoords( &block );
      blocksize = block.shape.a * block.shape.b * block.shape.c * block.shape.d * block.shape.e;
      
      // Map 5-D torus A-B-C-D-E to 3-D X-Y-Z cartesian space
      // by multiplying adjacent coordinates:
      // X = A * B
      // Y = C * D
      // Z = E * ranks per node
      //_nSlicesXYZ[0]=personality.Network_Config.Anodes * personality.Network_Config.Bnodes;
      //_nSlicesXYZ[1]=personality.Network_Config.Cnodes * personality.Network_Config.Dnodes;
      //_nSlicesXYZ[2]=personality.Network_Config.Enodes * Kernel_ProcessCount();
      
      _X=block.shape.a * block.shape.b;
      _Y=block.shape.c * block.shape.d;
      _Z=block.shape.e * Kernel_ProcessCount();
#else
      _runTimeTopologyExists=false;
#endif
#endif
#endif
      std::ifstream inFile("Topology.h", std::ios::in);
      if (inFile) {
	_storedTopologyExists=true;
	inFile.close();
      }

      if (READ_TOPOLOGY(_runTimeTopologyExists, _storedTopologyExists)) {
	int dim[3]={-1,-1,-1};
	bool topologyRead=true;
	std::ifstream inFile("Topology.h", std::ios::in);
	if (inFile) {
	  for (int i=0; i<3; ++i) {
	    std::string topologyLine;
	    getline (inFile, topologyLine);
	    if (topologyLine!="")  {
	      std::string str = topologyLine;
	      std::stringstream strstr(str);
	      std::istream_iterator<std::string> it(strstr);
	      std::istream_iterator<std::string> end;
	      std::vector<std::string> topologyResults(it, end);
	      if (topologyResults.size()!=3) {
		topologyRead=false;
		break;
	      }
	      dim[i]=atoi(topologyResults.at(2).c_str());
	    }
	    else {
	      topologyRead=false;
	      break;
	    }
	  }	  
	  inFile.close();
	}
	if (topologyRead) {
	  _X=dim[0];
	  _Y=dim[1];
	  _Z=dim[2];
	  _runTimeTopologyExists=true;
	}
      }
      else {
	std::ofstream outFile;
	outFile.open("Topology.h", std::ios::out);
	assert(outFile);
	outFile<<"#define _X_ "<<_X<<std::endl;
	outFile<<"#define _Y_ "<<_Y<<std::endl;
	outFile<<"#define _Z_ "<<_Z<<std::endl;
	outFile.close();
      }
    }
    
    int getX() {return _X;}
    int getY() {return _Y;}
    int getZ() {return _Z;}

    bool runTimeExists() {return _runTimeTopologyExists;}
    bool storedExists() {return _storedTopologyExists;}
private :
    int _X;
    int _Y;
    int _Z;
    bool _runTimeTopologyExists, _storedTopologyExists;
};
#endif
