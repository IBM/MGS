// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-2012
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

/*
 * NeurogenBranch.cpp
 *
 *  Created on: Jul 5, 2012
 *      Author: heraldo
 */

#include "NeurogenBranch.h"
#include "NeurogenSegment.h"

NeurogenBranch::NeurogenBranch() : soma(0)
{
}

NeurogenBranch::~NeurogenBranch()
{
}

int NeurogenBranch::getNrSegments()
{
  return BSegments.size();
}

void NeurogenBranch::addSegment(NeurogenSegment * seg)
{
  BSegments.push_back(seg);
}

double NeurogenBranch::getDistanceFirstLastSegment()
{
  double length = 0;
  if (getNrSegments()>0) {
    // return sqrt(SqDist(_segments[0].getCoords(), _segments[_numberOfSegments-1].getCoords()));
    length = sqrt( (BSegments[getNrSegments()-1]->getX() - BSegments[0]->getX()) * (BSegments[getNrSegments()-1]->getX() - BSegments[0]->getX()) +
		   (BSegments[getNrSegments()-1]->getY() - BSegments[0]->getY()) * (BSegments[getNrSegments()-1]->getY() - BSegments[0]->getY()) +
		   (BSegments[getNrSegments()-1]->getZ() - BSegments[0]->getZ()) * (BSegments[getNrSegments()-1]->getZ() - BSegments[0]->getZ()) );
  }
  return length;
}

//double NeurogenBranch::getTotalSegmentLength()
double NeurogenBranch::getLength()
{
  double length = 0;
  for (unsigned int i=0; i<BSegments.size(); i++) {
    length+=BSegments[i]->getLength();
  }
  return length;
}

NeurogenSegment* NeurogenBranch::getLastSegment()
{
  return BSegments[getNrSegments()-1];
}

NeurogenSegment* NeurogenBranch::getFirstSegment()
{
  return BSegments[0];
}

void NeurogenBranch::setSegmentIDs(NeurogenParams* params_p, int& count)
{
  for (unsigned int j=0; j<BSegments.size(); j++) {
    BSegments[j]->setID(count);
    BSegments[j]->setParentID(BSegments[j]->getParentSeg()->getID());
    count++;
  }
}

void NeurogenBranch::writeToSWC(std::ofstream& fout)
{
  for (int j=0; j<BSegments.size(); j++) {
    fout << BSegments[j]->outputLine();
    fout << std::endl;
  }
}  


bool NeurogenBranch::isBranching()
{
  return (Waypoint1.size()>0 && Waypoint2.size()>0);
}

void NeurogenBranch::addWaypoint(ShallowArray<double>& wp)
{
  assert(!isBranching());
  if (Waypoint1.size()>0) Waypoint2=wp;
  else Waypoint1=wp;
}


void NeurogenBranch::reset()
{
  for (unsigned int i=0; i<BSegments.size(); i++) {
    BSegments[i]->setNeuriteOrigin(false);
  }
  clearWaypoints();
  BSegments.clear();  
}
