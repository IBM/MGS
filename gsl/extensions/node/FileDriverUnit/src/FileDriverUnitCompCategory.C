// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "FileDriverUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_FileDriverUnitCompCategory.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <iterator>

#define RANK getSimulation().getRank()
#define ITER getSimulation().getIteration()
#define SHD getSharedMembers()

FileDriverUnitCompCategory::FileDriverUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
  : CG_FileDriverUnitCompCategory(sim, modelName, ndpList)
{
}

void FileDriverUnitCompCategory::initializeShared(RNG& rng) 
{
  // Setup period to load from file
  SHD.period = (unsigned) ((1.0 / SHD.deltaT) / (1.0 /SHD.sf));
  SHD.userWarned = false;
  
  // Setup input file stream
  std::ostringstream os;
  os.str("");
  os << SHD.inputFileName;
  ifs = new std::ifstream(os.str().c_str());
  *ifs >> SHD.total_time >> SHD.n_channels;
  line = new std::string();
  
  // Setup input array
  SHD.input.increaseSizeTo(SHD.n_channels);
  for (int i=0; i < SHD.n_channels; i++)
    SHD.input[i] = 0.0;
  
  // Section stuff, include if needed
  /*
  // set offset to beginning of section of observations if section is specified
  if (SHD.section == 0)
  {
  SHD.section = 1;
  } 
  else
  {
  std::cout << "here" << std::endl;

  std::getline(*ifs, *line); 
  std::getline(*ifs, *line); 
  SHD.offset+=2; //set to beginning of observations
  for(int current_section=1; current_section<SHD.section; SHD.offset++){
  std::getline(*ifs, *line);
  if (line->empty())
  current_section++;
  }
  }
  */
}


void FileDriverUnitCompCategory::readInputFile(RNG& rng) {
  if (ITER % SHD.period == 0)
    {
      // Check if at end of file
      if ((ITER >= SHD.total_time * SHD.period) && (!SHD.userWarned))
        {
          std::cerr << "Error: Reached end of input file." << std::endl;
          SHD.userWarned = true;
        }
      
      // Read next line and parse in to a string vector
      std::getline(*ifs, *line);
      std::istringstream iss(*line);
      std::vector<std::string> vals;
      copy(std::istream_iterator<std::string>(iss),
           std::istream_iterator<std::string>(),
           std::back_inserter(vals));

      // Check/handle for new section and/or errors
      if (vals.empty())
        {
          vals = std::vector<std::string>(SHD.n_channels, "0.0");
          //          std::cout << "Section " << SHD.section << " finished." << std::endl;
          //          if (RANK==0) { SHD.section++; } 
        }
      else if (vals.size() != SHD.n_channels)
        {
          std::cerr << "Error: Number of channels inconsistent." << std::endl;
          exit(1);
        }

      // Parse string vector in to double array
      for (int i=0; i < SHD.n_channels; i++)
        SHD.input[i] = std::strtod(vals[i].c_str(), NULL);

      // Section stuff, include if needed
      /*
      // transfer input value to individual nodes 
      int i=0;
      ShallowArray<FileDriverUnit>::iterator it = _nodes.begin();
      for (it; it<=_nodes.end() && i<SHD.n_channels; it++)
        {
          it->readInput(std::strtod(vals[i].c_str(), NULL));
          i++;
        }
      SHD.offset++;
      line->clear();
      //fs.close();
      */
    }
}
