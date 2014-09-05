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
 * NeurogenParams.cpp
 *
 *  Created on: Jun 25, 2012
 *      Author: heraldo
 */

#include "NeurogenParams.h"
#include <math.h>


NeurogenParams::NeurogenParams(int rank) 
  :  startX(0),              // X-coordinate of the very first point of the Soma
     startY(0),              //  Y-coordinate of the very first point of the Soma
     startZ(0),              // Z-coordinate of the very first point of the Soma
     nrStems(16),         // Nr of Stem branches that will start from the soma
     somaSurface(225),
     genZ(1.0),               // scaling factor for growth in Z dimension
     somaSegments(0),
     // important growth parameters
     startRadius(1.5),       // in microns, the starting Radius of all the Stem Segments when they are "born" from the Soma.
     umnPerFront(2.5),        // when n=2.0, the surface area of the each cylindrical segment, when n=1.0, the fixed length of each cylindrical segment
     nexpPerFront(0.0),      // n exponent used in computing umnPerFront minus 1.0
     radiusRate(0.995),       // the constant by which the Radius is multiplied on every growthStep.
     minRadius(0.2),
     RallsRatio(1.0),
     branchDiameterAsymmetry(0), // determines how long an asymmetery in branch diameters persists in tree
     branchProb(0.04),        // A constant branching Probability which gives us direct control on how "branched" out the neuron will turn out.
     minBifurcationAngle(25),
     maxBifurcationAngle(45),// Max initial branching angle, when creating Stems
     minInitialStemAngle(180),
     somaRepulsion(1.0),
     somaDistanceExp(1.0),
     homotypicRepulsion(1.0),
     homotypicDistanceExp(2.0),
     boundaryRepulsion(1.0),
     boundaryDistanceExp(2.0),
     intolerance(0),           // in microns, the distance to a tissue boundary point within which a growing branch terminate
     forwardBias(1.0),
     waypointGenerator("NULL"),
     waypointAttraction(0.0),
     waypointDistanceExp(1.0),
     waypointExtent(10.0),

     // GLOBAL limiting parameters
     maxFiberLength(150000),   // in microns, is the total Fiber Length available to the neuron, NeuroGeneration stops when it reaches that length
     width(250000),              // Width size in microns of limiting Bounding Cuboid, the neuron cannot grow further than this. Branches get terminated when they touch the Cuboid Sides
     height(250000),             // Width size in microns of limiting Bounding Cuboid, the neuron cannot grow further than this. Branches get terminated when they touch the Cuboid Sides
     depth(250000),              // Width size in microns of limiting Bounding Cuboid, the neuron cannot grow further than this. Branches get terminated when they touch the Cuboid Sides
     maxBifurcations(INT_MAX),
     maxSegments(500000),       // not very important, used just in case we want to limit the total number of segments before it even reaches the maxFiberLength
     RandSeed(12345),        // Random Number Seed.
     gaussSD(0.25),
     maxResamples(30),
     boundingSurface(""),
     terminalField("NULL"),
     _rank(rank)
{
  _rng.reSeed(RandSeed, _rank);
  convertDegreesToRadians();
}

NeurogenParams::NeurogenParams(std::string fileName, int rank)
  :  startX(0),              // X-coordinate of the very first point of the Soma
     startY(0),              //  Y-coordinate of the very first point of the Soma
     startZ(0),              // Z-coordinate of the very first point of the Soma
     nrStems(16),            // Nr of stems that will start from the soma
     somaSurface(225),
     genZ(1.0),               // scaling factor for growth in Z dimension
     somaSegments(0),
  // important growth parameters
     startRadius(1.5),       // in microns, the starting Radius of all the Stem Segments when they are "born" from the Soma.
     umnPerFront(2.5),        // when n=2.0, the surface area of the each cylindrical segment, when n=1.0, the fixed length of each cylindrical segment
     nexpPerFront(0.0),      // n exponent used in computing umnPerFront minus 1.0
     radiusRate(0.995),       // the constant by which the Radius is multiplied on every growthStep.
     minRadius(0.2),
     RallsRatio(1.0),
     branchDiameterAsymmetry(0), // determines how long an asymmetery in branch diameters persists in tree
     branchProb(0.04),        // A constant branching Probability which gives us direct control on how "branched" out the neuron will turn out.
     minBifurcationAngle(25),
     maxBifurcationAngle(45),// Max initial branching angle, when creating Stems
     minInitialStemAngle(180),
     somaRepulsion(1.0),
     somaDistanceExp(1.0),
     homotypicRepulsion(1.0),
     homotypicDistanceExp(2.0),
     boundaryRepulsion(1.0),
     boundaryDistanceExp(2.0),
     intolerance(0),           // in microns, the distance to a tissue boundary point within which a growing branch terminate
     forwardBias(1.0),
     waypointGenerator("NULL"),
     waypointAttraction(0.0),
     waypointDistanceExp(1.0),
     waypointExtent(10.0),

  // GLOBAL limiting parameters
     maxFiberLength(150000),   // in microns, is the total Fiber Length available to the neuron, NeuroGeneration stops when it reaches that length
     width(250000),              // Width size in microns of limiting Bounding Cuboid, the neuron cannot grow further than this. Branches get terminated when they touch the Cuboid Sides
     height(250000),             // Width size in microns of limiting Bounding Cuboid, the neuron cannot grow further than this. Branches get terminated when they touch the Cuboid Sides
     depth(250000),              // Width size in microns of limiting Bounding Cuboid, the neuron cannot grow further than this. Branches get terminated when they touch the Cuboid Sides
     maxBifurcations(INT_MAX),
     maxSegments(500000),       // not very important, used just in case we want to limit the total number of segments before it even reaches the maxFiberLength
     RandSeed(12345),        // Random Number Seed.
     gaussSD(0.25),
     maxResamples(30),
     boundingSurface("surface_mesh.vtk"),
     terminalField("NULL"),
     _rank(rank)
{
  load(fileName, rank);
}

void NeurogenParams::load(std::string fileName, int rank)
{
  std::string line;
  std::ifstream myfile (fileName.c_str());
  int nrParLines = 0;
  if (myfile.is_open()) {
    while ( myfile.good() ) {
      getline (myfile, line);
      //std::cout << line << std::endl;
      if (line!="" && line.at(0)!='#') nrParLines++;
    }
    //std::cout << "Nr of Parameter Lines read from input file: " << nrParLines << std::endl;
  }
  myfile.clear();
  myfile.seekg(0, std::ios_base::beg);
  std::string* parLines=new std::string[nrParLines];
  int i=0;
  if (myfile.is_open()) {
    while (myfile.good() && i<nrParLines) {
      getline (myfile,line);
      // std::cout << line << std::endl;
      if (line.at(0)!='#') {
	parLines[i] = line;
	i++;
      }     
    }
    //std::cout << parLines[0] << std::endl;
    //std::cout << "Done Reading Parameter FIle" << std::endl;
  }
  myfile.clear();
  myfile.close();
  
  // Heraldo's   NeuroGeneration Parameters as of July 5th
  if (nrParLines>=41) {
    int n=-1;

    startX = atof(parLines[++n].c_str());             // X-coordinate of the very first point of the Soma
    startY = atof(parLines[++n].c_str());             //  Y-coordinate of the very first point of the Soma
    startZ = atof(parLines[++n].c_str());             // Z-coordinate of the very first point of the Soma
    // Starting point of neuron generation
    nrStems = atoi(parLines[++n].c_str());        // Nr of Stem branches that will start from the soma

    somaSurface = atof(parLines[++n].c_str());
    genZ = atof(parLines[++n].c_str());
    somaSegments = atoi(parLines[++n].c_str()); 
    // important growth parameters
    startRadius = atof(parLines[++n].c_str());        // In microns, the starting Radius of all the Stem Segments when they are "born" from the Soma
    umnPerFront = atof(parLines[++n].c_str());        // when n=2.0, the surface area of the each cylindrical segment, when n=1.0, the fixed length of each cylindrical segment

    nexpPerFront = atof(parLines[++n].c_str());       // n exponent used in computing umnPerFront minus 1.0
    radiusRate = atof(parLines[++n].c_str());;        // the constant by which the Radius is multiplied on every growthStep.
    minRadius = atof(parLines[++n].c_str());    
    RallsRatio = atof(parLines[++n].c_str());
    branchDiameterAsymmetry = atof(parLines[++n].c_str()); // determines how long an asymmetery in branch diameters persists in tree
    branchProb = atof(parLines[++n].c_str());        // A constant branching Probability which gives us direct control on how "branched" out the neuron will turn out.

    minBifurcationAngle = atof(parLines[++n].c_str());
    maxBifurcationAngle = atof(parLines[++n].c_str());// Max initial branching angle, when creating Stems
    minInitialStemAngle = atof(parLines[++n].c_str());

    somaRepulsion = atof(parLines[++n].c_str());
    somaDistanceExp = atof(parLines[++n].c_str());

    homotypicRepulsion = atof(parLines[++n].c_str());
    homotypicDistanceExp = atof(parLines[++n].c_str());

    boundaryRepulsion = atof(parLines[++n].c_str());
    boundaryDistanceExp = atof(parLines[++n].c_str());

    intolerance = atof(parLines[++n].c_str());       // in microns, the distance to a tissue boundary point within which a growing branch terminate
    forwardBias = atof(parLines[++n].c_str());

    // GLOBAL limiting parameters
    maxFiberLength = atof(parLines[++n].c_str());    // in microns, is the total Fiber Length available to the neuron, NeuroGeneration stops when it reaches that length
    width = atof(parLines[++n].c_str());             // Width X size in microns of limiting Bounding Cuboid the neuron cannot grow further than this. Branches get terminated when they touch the Cuboid Sides
    height = atof(parLines[++n].c_str());            // Width X size in microns of limiting Bounding Cuboid the neuron cannot grow further than this. Branches get terminated when they touch the Cuboid Sides
    depth = atof(parLines[++n].c_str());             // Width X size in microns of limiting Bounding Cuboid the neuron cannot grow further than this. Branches get terminated when they touch the Cuboid Sides
    maxBifurcations = atoi(parLines[++n].c_str());

    maxSegments = atoi(parLines[++n].c_str());       // not very important, used just in case we want to limit the total number of segments before it even reaches the maxFiberLength
    RandSeed = atoi(parLines[++n].c_str());
    gaussSD = atof(parLines[++n].c_str());
    maxResamples = atof(parLines[++n].c_str());
    boundingSurface = parLines[++n].c_str();
    waypointGenerator = parLines[++n].c_str();
    waypointAttraction = atof(parLines[++n].c_str());
    waypointDistanceExp = atof(parLines[++n].c_str());
    waypointExtent = atof(parLines[++n].c_str());
    terminalField = parLines[++n].c_str();    

    delete [] parLines;
  }
  else if (fileName!="NULL") {
    std::cout << "Warning! Problem with the parameter file "<<fileName<<"! Using default parameters..." << std::endl;
  }
  _rng.reSeed(RandSeed, _rank);
  convertDegreesToRadians();
}

void NeurogenParams::printParams()
{
  std::string fileName = "ParamsOut";
  fileName += ".txt";
  std::ofstream fout;
  fout.open(fileName.c_str());
  printParams(fout, true, true, "#", "\n", "\n");
  fout.close();
}

void NeurogenParams::printParams(std::ostream& os,
				 bool names, bool values, 
				 const char* preamble, 
				 const char* namesSeparator, const char* valuesSeparator)
{
  if (names)  os << preamble << "start_X" << namesSeparator; 
  if (values) os << startX << valuesSeparator;
  if (names)  os << preamble << "start_Y" << namesSeparator; 
  if (values) os << startY << valuesSeparator;
  if (names)  os << preamble << "start_Z" << namesSeparator; 
  if (values) os << startZ << valuesSeparator;
  if (names)  os << preamble << "nbr_stems" << namesSeparator; 
  if (values) os << nrStems << valuesSeparator;
  if (names)  os << preamble << "soma_surface_area" << namesSeparator; 
  if (values) os << somaSurface << valuesSeparator;
  if (names)  os << preamble << "flatness" << namesSeparator; 
  if (values) os << genZ << valuesSeparator;
  if (names)  os << preamble << "soma_shape_segments" << namesSeparator; 
  if (values) os << somaSegments <<valuesSeparator;
  if (names)  os << preamble << "start_radius" << namesSeparator; 
  if (values) os << startRadius << valuesSeparator;
  if (names)  os << preamble << "front_extention_length" << namesSeparator; 
  if (values) os << umnPerFront << valuesSeparator; 
  if (names)  os << preamble << "front_extension_exp" << namesSeparator; 
  if (values) os << nexpPerFront << valuesSeparator;
  if (names)  os << preamble << "taper_rate" << namesSeparator; 
  if (values) os << radiusRate << valuesSeparator; 
  if (names)  os << preamble << "minimum_radius" << namesSeparator; 
  if (values) os << minRadius << valuesSeparator;  
  if (names)  os << preamble << "ralls_ratio" << namesSeparator; 
  if (values) os << RallsRatio << valuesSeparator; 
  if (names)  os << preamble << "branch_diameter_asymmetry" << namesSeparator;
  if (values) os << branchDiameterAsymmetry << valuesSeparator; 
  if (names)  os << preamble << "branch_probability" << namesSeparator; 
  if (values) os << branchProb<< valuesSeparator; 
  if (names)  os << preamble << "min_bifurcation_angle" << namesSeparator; 
  if (values) os << minBifurcationAngle*180.0/M_PI << valuesSeparator; 
  if (names)  os << preamble << "max_bifurcation_angle" << namesSeparator; 
  if (values) os << maxBifurcationAngle*180.0/M_PI << valuesSeparator; 
  if (names)  os << preamble << "min_initial_stem_angle" << namesSeparator; 
  if (values) os << minInitialStemAngle*180.0/M_PI << valuesSeparator; 
  if (names)  os << preamble << "soma_tropic_force" << namesSeparator; 
  if (values) os << somaRepulsion<< valuesSeparator; 
  if (names)  os << preamble << "soma_tropic_decay" << namesSeparator; 
  if (values) os << somaDistanceExp << valuesSeparator; 
  if (names)  os << preamble << "self_avoidance_force" << namesSeparator; 
  if (values) os << homotypicRepulsion << valuesSeparator; 
  if (names)  os << preamble << "self_avoidance_decay" << namesSeparator; 
  if (values) os << homotypicDistanceExp << valuesSeparator; 
  if (names)  os << preamble << "boundary_avoidance_force" << namesSeparator; 
  if (values) os << boundaryRepulsion << valuesSeparator; 
  if (names)  os << preamble << "boundary_avoidance_decay" << namesSeparator; 
  if (values) os << boundaryDistanceExp << valuesSeparator; 
  if (names)  os << preamble << "intersection_proximity" << namesSeparator; 
  if (values) os << intolerance << valuesSeparator; 
  if (names)  os << preamble << "inertial_force" << namesSeparator; 
  if (values) os << forwardBias << valuesSeparator; 
  if (names)  os << preamble << "max_fiber_length" << namesSeparator; 
  if (values) os << maxFiberLength << valuesSeparator;
  if (names)  os << preamble << "bounding_width" << namesSeparator; 
  if (values) os << width << valuesSeparator;  
  if (names)  os << preamble << "bounding_height" << namesSeparator; 
  if (values) os << height << valuesSeparator; 
  if (names)  os << preamble << "bounding_depth" << namesSeparator; 
  if (values) os << depth << valuesSeparator; 
  if (names)  os << preamble << "max_bifurcations" << namesSeparator; 
  if (values) os << maxBifurcations << valuesSeparator;
  if (names)  os << preamble << "max_segments" << namesSeparator; 
  if (values) os << maxSegments << valuesSeparator; 
  if (names)  os << preamble << "rand_seed" << namesSeparator; 
  if (values) os << RandSeed << valuesSeparator; 
  if (names)  os << preamble << "gauss_sd" << namesSeparator; 
  if (values) os << gaussSD << valuesSeparator;
  if (names)  os << preamble << "max_resamples" << namesSeparator; 
  if (values) os << maxResamples << valuesSeparator;
  if (names)  os << preamble << "bounding_mesh" << namesSeparator; 
  if (values) os << boundingSurface << valuesSeparator;
  if (names)  os << preamble << "waypoint_generator" << namesSeparator; 
  if (values) os << waypointGenerator << valuesSeparator;
  if (names)  os << preamble << "waypoint_attraction" << namesSeparator; 
  if (values) os << waypointAttraction << valuesSeparator;
  if (names)  os << preamble << "waypoint_decay" << namesSeparator; 
  if (values) os << waypointDistanceExp << valuesSeparator;
  if (names)  os << preamble << "waypoint_extent" << namesSeparator; 
  if (values) os << waypointExtent << valuesSeparator;
  if (names)  os << preamble << "terminal_field" << namesSeparator; 
  if (values) os << terminalField << valuesSeparator;
  os << std::endl;
}

double NeurogenParams::getGaussian(double mu, double sigma)
{
  double rval = gaussian(mu, sigma, _rng);
  return rval;
}

void NeurogenParams::convertDegreesToRadians()
{
  minBifurcationAngle=minBifurcationAngle*M_PI/180.0;
  maxBifurcationAngle=maxBifurcationAngle*M_PI/180.0;
  minInitialStemAngle=minInitialStemAngle*M_PI/180.0;
}
