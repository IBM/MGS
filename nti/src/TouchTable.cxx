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

#include "TouchTable.h"
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <float.h>

TouchTable::TouchTable(std::vector<SegmentDescriptor::SegmentKeyData> const & maskVector)
  : _output(true)
{	
  _mask = _segmentDescriptor.getMask(maskVector);
}

TouchTable::~TouchTable()
{
}

void TouchTable::evaluateTouch(double segKey1, double segKey2)
{
  double key1 = _segmentDescriptor.getSegmentKey(segKey1, _mask);
  double key2 = _segmentDescriptor.getSegmentKey(segKey2, _mask);

  std::map<double, std::map<double, long> >::iterator mapIter1;  	
  mapIter1 = _touchTable.find(key1);
  if (mapIter1==_touchTable.end()) {
    std::map<double, long> newMap;
    newMap[key2] = 1;
    assert(_touchTable.size()<_touchTable.max_size());
    _touchTable[key1] = newMap;
  }
  else {
    std::map<double, long>& map2 = mapIter1->second;
    std::map<double, long>::iterator mapIter2;  		
    mapIter2 = map2.find(key2);
    if (mapIter2 == map2.end()) {
      assert(map2.size()<map2.max_size());
      map2[key2] = 1;
    }
    else {
      mapIter2->second++;
    }
  }
}

void TouchTable::reset()
{
  std::map<double, std::map<double, long> >::iterator mapIter;
  std::map<double, std::map<double, long> >::iterator mapEnd = _touchTable.end();
  for (mapIter = _touchTable.begin(); mapIter != mapEnd; ++mapIter) {
    mapIter->second.clear();
  }
  _touchTable.clear();
}
	
int TouchTable::size() 
{
  int rval=0;
  std::map<double, std::map<double, long> >::iterator mapIter;
  std::map<double, std::map<double, long> >::iterator mapEnd = _touchTable.end();
  for (mapIter = _touchTable.begin(); mapIter != mapEnd; ++mapIter) {
    rval += mapIter->second.size();		
  }
  return rval;
}

int TouchTable::getTouchCount() 
{
  int rval=0;
  std::map<double, std::map<double, long> >::iterator mapIter;
  std::map<double, std::map<double, long> >::iterator mapEnd = _touchTable.end();
  for (mapIter = _touchTable.begin(); mapIter != mapEnd; ++mapIter) {
    std::map<double, long>::iterator mapIter2;
    std::map<double, long>::iterator mapEnd2 = mapIter->second.end();
    for (mapIter2 = mapIter->second.begin(); mapIter2 != mapEnd2; ++mapIter2) {
      rval += mapIter2->second;		
    }
  }
  return rval;
}

void TouchTable::getEntries(TouchTableEntry* outputBuffer)
{
  std::map<double, std::map<double, long> >::iterator mapIter1;
  std::map<double, std::map<double, long> >::iterator mapEnd1 = _touchTable.end();
  for (mapIter1 = _touchTable.begin(); mapIter1 != mapEnd1; ++mapIter1) {	
    std::map<double, long>& map2 = mapIter1->second;
    std::map<double, long>::iterator mapIter2;
    std::map<double, long>::iterator mapEnd2 = map2.end();
    double id1 = mapIter1->first;
    for (mapIter2 = map2.begin(); mapIter2!=mapEnd2; ++mapIter2) {
      outputBuffer->key1=id1;
      outputBuffer->key2=mapIter2->first;
      outputBuffer->count=mapIter2->second;
      ++outputBuffer;
    }
  }
}

void TouchTable::getEntries(TouchTableEntry* outputBuffer, 
			    int numberOfEntries,
			    std::map<double, std::map<double, long> >::iterator& mapIter1,
			    std::map<double, long>::iterator& mapIter2,
			    bool& complete)
{
  std::map<double, std::map<double, long> >::iterator mapEnd1 = _touchTable.end();
  std::map<double, long>::iterator mapEnd2;
  if (complete) {
    mapIter1 = _touchTable.begin();
    if (mapIter1 != mapEnd1) {
      mapEnd2 = mapIter1->second.end();
      mapIter2 = mapIter1->second.begin();
      complete=false;
    }
  }
  else mapEnd2 = mapIter1->second.end();

  if (!complete) {
    int count=0;
    while(1) {
      double id1 = mapIter1->first;
      do {
	outputBuffer->key1=id1;
	outputBuffer->key2=mapIter2->first;
	outputBuffer->count=mapIter2->second;
	++outputBuffer;
	++count;
	++mapIter2;
      } while (mapIter2 != mapEnd2 && count!=numberOfEntries);
      
      if (mapIter2==mapEnd2) {
	if (++mapIter1 == mapEnd1) break;
	mapIter2 = mapIter1->second.begin();
	mapEnd2 = mapIter1->second.end();
      }
      if (count==numberOfEntries) break;
    }    
    if (mapIter2==mapEnd2 && mapIter1==mapEnd1) complete=true;
  }
}

void TouchTable::writeToFile(int tableNumber, int iterationNumber, std::string experimentName)
{
  char filename[256];
  sprintf(filename,"outTable%d_%s_%d.bin", tableNumber, experimentName.c_str(),  iterationNumber);
  FILE* data;
  if((data = fopen(filename, "wb")) == NULL) {
    printf("Could not open the output file %s!\n", filename);
    MPI_Finalize();
    exit(0);
  }
  
  int sz=size();
  fwrite(&sz, sizeof(int), 1, data);
  std::map<double, std::map<double, long> >::iterator mapIter = _touchTable.begin();  	
  std::map<double, std::map<double, long> >::iterator mapEnd = _touchTable.end();  	
  for (; mapIter!=mapEnd; ++mapIter) {
    std::map<double, long>::iterator mapIter2 = mapIter->second.begin();  	
    std::map<double, long>::iterator mapEnd2 = mapIter->second.end();  	
    for (; mapIter2!=mapEnd2; ++mapIter2) {
      fwrite(&mapIter->first, sizeof(double), 1, data);
      fwrite(&mapIter2->first, sizeof(double), 1, data);
      fwrite(&mapIter2->second, sizeof(long), 1, data);
    }
  }
  fclose(data);
}

void TouchTable::outputTable(int tableNumber, int iterationNumber, std::string experimentName)
{
  if (_output) {
    printf("Table%d_%s_%d\n", tableNumber, experimentName.c_str(), iterationNumber);
    _segmentDescriptor.printMask(_mask);
    std::map<double, std::map<double, long> >::iterator mapIter = _touchTable.begin();  	
    std::map<double, std::map<double, long> >::iterator mapEnd = _touchTable.end();  	
    for (; mapIter!=mapEnd; ++mapIter) {
      std::map<double, long>::iterator mapIter2 = mapIter->second.begin();  	
      std::map<double, long>::iterator mapEnd2 = mapIter->second.end();  	
      for (; mapIter2!=mapEnd2; ++mapIter2) {
	_segmentDescriptor.printKey(mapIter->first, _mask);
	printf(" ->\n");
	_segmentDescriptor.printKey(mapIter2->first, _mask);
	printf(" : \t%ld\n\n", mapIter2->second);
      }
    }
    fflush(stdout);
  }
}


void TouchTable::addEntries(TouchTableEntry* buffer, TouchTableEntry* bufferEnd)
{
  std::map<double, std::map<double, long> >::iterator mapIter1;  	
  for (; buffer != bufferEnd; ++buffer) {

    double segKey1 = buffer->key1;
    double segKey2 = buffer->key2;
    long count = buffer->count;
		
    mapIter1 = _touchTable.find(segKey1);
    if (mapIter1==_touchTable.end()) {
      std::map<double, long> newMap;
      newMap[segKey2] = count;
      assert(_touchTable.size()<_touchTable.max_size());
      _touchTable[segKey1] = newMap;
    }
    else {
      std::map<double, long>& map2 = mapIter1->second;
      std::map<double, long>::iterator mapIter2;  		
      mapIter2 = map2.find(segKey2);
      if (mapIter2 == map2.end()) {
	assert(map2.size()<map2.max_size());
	map2[segKey2] = count;
      }
      else {
	mapIter2->second += count;
      }
    }
  }
}
