// =============================================================================
// (C) Copyright IBM Corp. 2005-2026. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
//
// Validation of untrusted tissue-spec and SWC morphology input.
//
// The text loaders (NeuronPartitioner::countAllNeurons, Tissue::loadText,
// Neuron::loadText, Branch::loadText) were written for well-formed
// NeuroMorpho-style SWC files and use SWC ids directly as array indices.
// These helpers state that contract explicitly and reject any file that
// breaks it, before a single Segment is allocated or written.

#ifndef SWCINPUT_H
#define SWCINPUT_H

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace SwcInput
{

// Number of cortical layers a tissue-spec LAYER field may name (0..5).
static const int MAX_LAYERS = 6;

[[noreturn]] inline void fail(const std::string& what, const char* file)
{
  std::cerr << "Invalid input in " << (file ? file : "(unknown)") << ": "
            << what << std::endl;
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::exit(EXIT_FAILURE);  // not reached
}

// LAYER indexes per-layer arrays of size MAX_LAYERS.
inline void checkLayer(int layer, const char* tissueSpec)
{
  if (layer < 0 || layer >= MAX_LAYERS)
    fail("tissue-spec LAYER " + std::to_string(layer) +
             " is outside 0.." + std::to_string(MAX_LAYERS - 1),
         tissueSpec);
}

// Reads every SWC record from the current position of 'fp' (comments must
// already be skipped), checks the structure the loader relies on, restores
// the file position, and returns the number of Segments the loader creates.
//
// Contract (all 25 SWC files shipped in the repository satisfy it):
//   1. every record has 7 numeric fields: id type x y z radius parent;
//   2. the first record is the single root (parent == -1, id == 1);
//   3. it may be followed by soma-continuation records (type 1, parent 1),
//      which are averaged into the soma and do not become Segments;
//   4. every later record has id == its position (ids are sequential after
//      removing the continuation records) and a parent that is an earlier
//      record (1 <= parent < id), because ids are used as array indices.
// Any violation aborts with a message naming the file and record.
inline int countSegments(FILE* fp, const char* swcFile)
{
  const long start = std::ftell(fp);
  int id, type, parent;
  double x, y, z, r;
  long long records = 0, continuation = 0;
  bool inSomaBlock = true;
  int n;
  while ((n = std::fscanf(fp, "%d %d %lf %lf %lf %lf %d", &id, &type, &x, &y,
                          &z, &r, &parent)) == 7)
  {
    ++records;
    if (records > 0x7fffffffLL) fail("too many SWC records", swcFile);
    const std::string where = "record " + std::to_string(records) + " (id " +
                              std::to_string(id) + ")";
    if (records == 1)
    {
      if (parent != -1 || id != 1)
        fail(where + ": the first record must be the root (id 1, parent -1)",
             swcFile);
      continue;
    }
    if (inSomaBlock && type == 1 && parent == 1)
    {
      ++continuation;
      continue;
    }
    inSomaBlock = false;
    if (parent == -1)
      fail(where + ": only one root record (parent -1) is supported", swcFile);
    // Ids as the loader sees them, after dropping continuation records.
    const long long expected = records - continuation;
    const long long seg = (long long)id - continuation;
    const long long par =
        (parent == 1) ? 1 : (long long)parent - continuation;
    if (seg != expected)
      fail(where + ": ids must be sequential (expected id " +
               std::to_string(expected + continuation) + ")",
           swcFile);
    if (par < 1 || par >= seg)
      fail(where + ": parent " + std::to_string(parent) +
               " must name an earlier record",
           swcFile);
  }
  if (n != EOF)
    fail("record " + std::to_string(records + 1) +
             " is malformed (expected 7 numeric fields)",
         swcFile);
  if (records == 0) fail("no SWC records", swcFile);
  std::fseek(fp, start, SEEK_SET);
  return (int)(records - continuation);
}

}  // namespace SwcInput

#endif
