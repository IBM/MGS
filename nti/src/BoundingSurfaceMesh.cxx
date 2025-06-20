// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "BoundingSurfaceMesh.h"
#include "NeurogenParams.h"
#include "NeurogenSegment.h"

#include <sstream>
#include <float.h>
#include <math.h>

BoundingSurfaceMesh::BoundingSurfaceMesh(std::string filename)
    : _hx(0),
      _hy(0),
      _hz(0),
      _npts(0),
      _ntriangles(0),
      _A(0),
      _B(0),
      _C(0),
      _norms(0),
      _distPtsSqrd(0),
      _distTrgSqrd(0),
      _meanX(0),
      _meanY(0),
      _meanZ(0),
      _minDistSqrd(DBL_MAX),
      boundless(false) {
  FILE* inputFile;
  std::ostringstream os;
  os << filename;
  std::stringstream ss;
  std::string line;
  std::string tok;
  std::ifstream myfile(os.str().c_str());
  if (myfile.is_open()) {
    getline(myfile, line);
    getline(myfile, line);
    getline(myfile, line);
    getline(myfile, line);
    getline(myfile, line);
    ss.clear();
    ss.str("");
    ss << line;
    ss >> tok;
    assert(tok == "POINTS");
    ss >> _npts;
    ss >> tok;
    assert(tok == "float");
    _hx = new double[_npts];
    _hy = new double[_npts];
    _hz = new double[_npts];
    _distPtsSqrd = new double[_npts];
    for (int i = 0; i < _npts;) {
      getline(myfile, line);
      ss.clear();
      ss.str("");
      ss << line;
      while (1) {
        ss >> _hx[i] >> _hy[i] >> _hz[i];
        if (ss.fail()) break;
        _meanX += _hx[i];
        _meanY += _hy[i];
        _meanZ += _hz[i];
        ++i;
      }
    }
    _meanX /= _npts;
    _meanY /= _npts;
    _meanZ /= _npts;
    for (int i = 0; i < _npts; ++i) {
      _hx[i] -= _meanX;
      _hy[i] -= _meanY;
      _hz[i] -= _meanZ;
      if ((_distPtsSqrd[i] = _hx[i] * _hx[i] + _hy[i] + _hy[i] +
                             _hz[i] * _hz[i]) < _minDistSqrd)
        _minDistSqrd = _distPtsSqrd[i];
    }
    getline(myfile, line);
    ss.clear();
    ss.str("");
    ss << line;
    ss >> tok;
    assert(tok == "POLYGONS");
    int nints;
    ss >> _ntriangles >> nints;
    _A = new int[_ntriangles];
    _B = new int[_ntriangles];
    _C = new int[_ntriangles];
    _norms = new double[_ntriangles * 3];

    _distTrgSqrd = new double[_ntriangles];
    int three;
    for (int i = 0; i < _ntriangles; ++i) {
      getline(myfile, line);
      ss.clear();
      ss.str("");
      ss << line;
      ss >> three;
      assert(three == 3);
      ss >> _A[i] >> _B[i] >> _C[i];

      double a[3] = {_hx[_A[i]], _hy[_A[i]], _hz[_A[i]]};
      double b[3] = {_hx[_B[i]], _hy[_B[i]], _hz[_B[i]]};
      double c[3] = {_hx[_C[i]], _hy[_C[i]], _hz[_C[i]]};
      double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
      double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
      double n[3];
      NeurogenSegment::cross(ab, ac, n);

      std::copy(n, n + 3, &_norms[i * 3]);
      // memcpy(&_norms[i*3], n, sizeof(double)*3);
      _distTrgSqrd[i] = _distPtsSqrd[_A[i]];
      if (_distPtsSqrd[_B[i]] < _distTrgSqrd[i])
        _distTrgSqrd[i] = _distPtsSqrd[_B[i]];
      if (_distPtsSqrd[_C[i]] < _distTrgSqrd[i])
        _distTrgSqrd[i] = _distPtsSqrd[_C[i]];
    }
    myfile.close();
  } else
    boundless = true;
}

bool BoundingSurfaceMesh::isOutsideVolume(NeurogenSegment* _seg) {
  bool rval = false;
  if (!boundless) {
    double p[3] = {_seg->getX() - _meanX, _seg->getY() - _meanY,
                   _seg->getZ() - _meanZ};
    double segDist = 0;
    if ((segDist = p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) > _minDistSqrd) {
      double q[3] = {0, 0, 0};
      for (int i = 0; i < _ntriangles; ++i) {
        if (segDist > _distTrgSqrd[i]) {
          double a[3] = {_hx[_A[i]], _hy[_A[i]], _hz[_A[i]]};
          double b[3] = {_hx[_B[i]], _hy[_B[i]], _hz[_B[i]]};
          double c[3] = {_hx[_C[i]], _hy[_C[i]], _hz[_C[i]]};
          double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
          double n[3];
          std::copy(&_norms[i * 3], &_norms[i * 3] + 3, n);
          // memcpy(n, &_norms[i * 3], sizeof(double) * 3);

          if (n[2] <= 0) {
            double tmp[3];
            std::copy(a, a + 3, tmp);
            // memcpy(tmp, a, sizeof(double) * 3);
            std::copy(b, b + 3, a);
            // memcpy(a, b, sizeof(double) * 3);
            std::copy(tmp, tmp + 3, b);
            // memcpy(b, tmp, sizeof(double) * 3);
            n[0] = -n[0];
            n[1] = -n[1];
            n[2] = -n[2];
          }

          double nap = _seg->dot(n, ap);
          if (nap != 0) {
            double aq[3] = {q[0] - a[0], q[1] - a[1], q[2] - a[2]};
            double naq = _seg->dot(n, aq);
            if (nap * naq < 0) {
              double tr = nap / (nap - naq);
              double r[3] = {p[0] + tr * (q[0] - p[0]),
                             p[1] + tr * (q[1] - p[1]),
                             p[2] + tr * (q[2] - p[2])};
              double abr = ((r[0] - b[0]) * (a[1] - b[1])) -
                           ((a[0] - b[0]) * (r[1] - b[1]));
              double bcr = ((r[0] - c[0]) * (b[1] - c[1])) -
                           ((b[0] - c[0]) * (r[1] - c[1]));
              double car = ((r[0] - a[0]) * (c[1] - a[1])) -
                           ((c[0] - a[0]) * (r[1] - a[1]));
              bool b0 = (abr > 0) ? true : false;
              bool b1 = (bcr > 0) ? true : false;
              bool b2 = (car > 0) ? true : false;
              if (b0 == b1 && b1 == b2) {
                rval = true;
                // assert(0);
                break;
              }
            }
          }
        }
      }
    }
  }
  return rval;
}

BoundingSurfaceMesh::~BoundingSurfaceMesh() {
  delete[] _hx;
  delete[] _hy;
  delete[] _hz;
  delete[] _A;
  delete[] _B;
  delete[] _C;
  delete[] _norms;
  delete[] _distPtsSqrd;
  delete[] _distTrgSqrd;
}
