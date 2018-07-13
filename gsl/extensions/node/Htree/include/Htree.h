#ifndef Htree_H
#define Htree_H

#include "CG_Htree.h"
#include "Lens.h"
#include "NVUMacros.h" // nvu index macros
#include "matrixOperations.h"
#include "rndm.h"
#include <fstream>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <sys/stat.h>

#define FILENAMESIZE 2014

#define POW_OF_2(x) (1 << (x)) // macro for 2^x using bitwise shift
#define UC_OLD_CODE 0 
#define IBM_NEW_CODE 1
//#define HTREE_IO_CHOICE UC_OLD_CODE 
#define HTREE_IO_CHOICE IBM_NEW_CODE

struct TreeWorkspace {
  // Geometrical information
  int N;    // Total number of levels */
  int Nsub; // Subtree size for blk-diagonal Jacobian */
  int N0;   // Number of root levels */
  int Np; // Number of levels for local subtree, e.g. if H-tree has 4 levels and
          // using 4 cores the local subtrees will have 2 levels */

  // First, the pieces required to describe the parallel tree problem
  cs *A;      // Local (per node) adjacency matrix
  cs *At;     // transpose of A
  cs *G;      // Conductance matrix
  int *level; // Level of each vessel in A
  double *g;  // Conductance, one per vessel. will be a pointer to data in G

  int nu;      // Number of equations total (NEQ * numberNVUs)
  int neq;     // Number of equations per block
  int mlocal;  // Size of subtree
  int nlocal;  // ..
  int mglobal; // Processor grid, number of rows
  int nglobal; // Processor grid, number of columns
  int ntimestamps;
  int QglobalPos;
  int PglobalPos;

  // Root subtree
  // cs      *A0;    // Root adjacency matrix
  // cs      *A0t;
  // int     *level0;// Level of each vessel in A0
  // cs      *G0;    // Root conductance matrix
  // double  *g0;    // Conductance vector

  // Vectors to hold local pressures and flows and solutions */
  css *symbchol; // symbolic Cholesky factorisation
  // css     *symbchol0; // factorisation of root
  double *l; // Scaled vessel lengths for regulating vessels
  double *b; // RHS vector
  // double  *b0;    // RHS vector for root
  double *p; // Pressure at each node (one per row of A)
  double *q; // Flow through each vessel
  double *w; // Pressure drop over each vessel
  // double  *p0;    // Pressures at root
  // double  *q0;    // Flow at root
  double *u; // Intermediate variable for parallel flow computation
  double *v; // Intermediate variable for parallel flow computation

  double *x; // x coordinates of NVU nodes
  double *y; // y coordinates of NVU nodes

  // double  *xn;    // extra n-sized workspace
  double *xm; // extra m-sized workspace
  // double  *xn0;   // extra n0-sized workspace

  // MPI Information
  int n_writes;
  int displacement;           // global displacement in output file, in bytes
  int displacement_per_write; // bytes written per write (globally)
  // double  *buf;                   // Communication buffer
  char *Toutfilename;
  char *Qoutfilename;
  char *Poutfilename;
  // MPI_File Toutfile;
  // MPI_File Qoutfile;
  // MPI_File Poutfile;
  char *dirName;
  MPI_Datatype subarray;
  MPI_Datatype subarray_single;

  double *stateVars; // array of all statevariables.

#if HTREE_IO_CHOICE == IBM_NEW_CODE
  long counter; // now keep how many data point written out
#elif HTREE_IO_CHOICE == UC_OLD_CODE
  int counter; //original is used to triggering I/O
#endif

  int *NVUIndices;
  int numberNVUs;
};

class Htree : public CG_Htree {
public:
  void initTree(RNG &rng);
  void updatePressures(RNG &rng);
  void finalize(RNG &rng);
  void copy(RNG &rng);
  void writeToFiles(RNG &rng);
  //virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);

  virtual void setPointers(const String &CG_direction,
                           const String &CG_component, NodeDescriptor *CG_node,
                           Edge *CG_edge, VariableDescriptor *CG_variable,
                           Constant *CG_constant,
                           CG_HtreeInAttrPSet *CG_inAttrPset,
                           CG_HtreeOutAttrPSet *CG_outAttrPset);
  virtual ~Htree();

private:
  std::ofstream *pressureOut;
  std::ofstream *flowOut;
  std::ofstream *dataOut;

  TreeWorkspace *tree;

  // void copyPressures();

  // Time-varying input pressure function
  double inputPressure(double t);
  void init_subtree();
  // void init_roottree();

  cs *adjacency(int Np);
  // The function that actually solves for p, w and q!
  void compute_sub(double p0, double pcap);
  void compute_uv(double pcap);
  void set_conductance(int unscaled);
  void compute_symbol_cholesky();
  void set_spatial_coordinates();

  /* lengths are in unit of meter */
  void set_length();
  double compute_length(int level, int n_levels);
  double compute_radius(int level, int n_levels);
  void malloc_tree();
  void init_io();
  void close_io();
#if HTREE_IO_CHOICE == IBM_NEW_CODE
  void write_data(int part_index=1);
  void write_info();
#elif HTREE_IO_CHOICE == UC_OLD_CODE
  void write_data();
  void write_info_n_data(); /* old version */
#endif
  void write_flow();
  void write_pressure();
  void calculate_indices();
  int _numberNVUs = 0;
  double _prevWritingTime = 0.0; /* the time (in sec) at which data is written*/
};

#endif
