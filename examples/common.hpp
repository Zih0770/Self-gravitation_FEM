#pragma once

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#include <iomanip>
#include <filesystem>

using namespace mfem;

class Nondimensionalisation {
private:
    real_t L;   // Length scale [m]
    real_t T;   // Time scale [s]
    real_t RHO; // Density scale [kg/m^3]

public:
    Nondimensionalisation(real_t length_scale, real_t time_scale, real_t density_scale)
        : L(length_scale), T(time_scale), RHO(density_scale) {}

    real_t Length() const { return L; }
    real_t Time() const { return T; }
    real_t Density() const { return RHO; }

    real_t Velocity() const { return L / T; }
    real_t Acceleration() const { return L / (T*T); }
    real_t Pressure() const { return RHO * L*L / (T*T); } 
    real_t Gravity() const { return L / (T*T); }
    real_t Potential() const { return L*L / (T*T); }

    real_t ScaleLength(real_t x) const { return x / L; }
    real_t UnscaleLength(real_t x_nd) const { return x_nd * L; }

    real_t ScaleDensity(real_t rho) const { return rho / RHO; }
    real_t UnscaleDensity(real_t rho_nd) const { return rho_nd * RHO; }

    real_t ScaleGravityPotential(real_t phi) const { return phi / Potential(); }
    real_t UnscaleGravityPotential(real_t phi_nd) const { return phi_nd * Potential(); }

    real_t ScaleStress(real_t sigma) const { return sigma / Pressure(); }
    real_t UnscaleStress(real_t sigma_nd) const { return sigma_nd * Pressure(); }

    void UnscaleGravityPotential(GridFunction &phi_gf) const { phi_gf *= Potential(); }
    void UnscaleDisplacement(GridFunction &u_gf) const { u_gf *= L; }
    void UnscaleStress(GridFunction &sigma_gf) const { sigma_gf *= Pressure(); }

    Coefficient *MakeScaledDensityCoefficient(Coefficient &rho_coeff) const {
        return new ProductCoefficient(1.0 / RHO, rho_coeff);
    }

    void Print() const {
        std::cout << "Scaling parameters:\n";
        std::cout << "  Length scale: " << L << " m\n";
        std::cout << "  Time scale: " << T << " s\n";
        std::cout << "  Density scale: " << RHO << " kg/m^3\n";
        std::cout << "  Gravity potential scale: " << Potential() << " m^2/s^2\n";
    }
};

static double EvalCoefficientAtPointGlobal(Coefficient &coef,
                                           ParMesh &pmesh,
                                           const Vector &x)
{
  double loc_val = 0.0; int loc_has = 0;
  for (int e = 0; e < pmesh.GetNE(); ++e) {
    ElementTransformation *T = pmesh.GetElementTransformation(e);
    InverseElementTransformation inv(T);
    IntegrationPoint ip;
    if (inv.Transform(x, ip) == InverseElementTransformation::Inside) {
      T->SetIntPoint(&ip);
      loc_val = coef.Eval(*T, ip);
      loc_has = 1;
      break;
    }
  }
  double glob_val = 0.0; int glob_has = 0;
  MPI_Allreduce(&loc_val, &glob_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&loc_has, &glob_has, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (glob_has > 0) { glob_val /= glob_has; }
  return glob_val;
}

class UniformSphereSolution {
 private:
  static constexpr mfem::real_t pi = std::atan(1) * 4;

  int _dim;
  mfem::real_t _r;

  mfem::Vector _x;

 public:
  UniformSphereSolution(int dim, const mfem::Vector& x, mfem::real_t r)
      : _dim{dim}, _x{x}, _r{r} {}

  mfem::FunctionCoefficient Coefficient() const {
    using namespace mfem;
    if (_dim == 2) {
      return FunctionCoefficient([this](const Vector& x) {
        auto r = x.DistanceTo(_x);
        if (r <= _r) {
          return pi * r * r;
        } else {
          return 2 * pi * _r * log(r / _r) + pi * _r * _r;
        }
      });
    } else {
      return FunctionCoefficient([this](const Vector& x) {
        auto r = x.DistanceTo(_x);
        if (r <= _r) {
          return -2 * pi * (3 * _r * _r - r * r) / 3;
        } else {
          return -4 * pi * pow(_r, 3) / (3 * r);
        }
      });
    }
  }

  mfem::FunctionCoefficient LinearisedCoefficient(const mfem::Vector& a) const {
    using namespace mfem;

    if (_dim == 2) {
      return FunctionCoefficient([this, a](const Vector& x) {
        auto dim = x.Size();
        auto r = x.DistanceTo(_x);
        auto dr = x;
        dr -= _x;
        dr /= r;
        if (r <= _r) {
          return -2 * pi * r * (dr * a);
        } else {
          return -2 * pi * (dr * a) / r;
        }
      });
    } else {
      return FunctionCoefficient([this, &a](const Vector& x) {
        auto dim = x.Size();
        auto r = x.DistanceTo(_x);
        auto dr = x;
        dr -= _x;
        dr /= r;
        if (r <= _r) {
          return -4 * pi * r * (dr * a) / 3;
        } else {
          return -4 * pi * std::pow(_r, 3) * (dr * a) / (3 * r * r);
        }
      });
    }
  }
};

class LonLatGrid
{
public:
  LonLatGrid(std::vector<double> lons, std::vector<double> lats)
      : _lons(std::move(lons)),
        _lats(std::move(lats)),
        _nlon(static_cast<int>(_lons.size())),
        _nlat(static_cast<int>(_lats.size()))
  {
  }

  int NLon() const { return _nlon; }
  int NLat() const { return _nlat; }

  const std::vector<double> &Lons() const { return _lons; }
  const std::vector<double> &Lats() const { return _lats; }

  double LonAt(int i) const { return _lons[i]; }
  double LatAt(int j) const { return _lats[j]; }

  size_t Idx(int i, int j) const
  {
    return static_cast<size_t>(j) * static_cast<size_t>(_nlon) +
           static_cast<size_t>(i);
  }

  double NorthPole(const std::vector<double> &field) const
  {
    const int j0 = _nlat - 2, j1 = _nlat - 1;
    const double y0 = _lats[j0], y1 = _lats[j1];
    const double dy = y1 - y0;
    const double t = (90.0 - y1) / dy + 1.0;

    double sum = 0.0;
    for (int i = 0; i < _nlon; ++i)
    {
      const double v0 = field[Idx(i, j0)];
      const double v1 = field[Idx(i, j1)];
      sum += v0 * (1.0 - t) + v1 * t;
    }
    return sum / _nlon;
  }

  double SouthPole(const std::vector<double> &field) const
  {
    const int j0 = 0, j1 = 1;
    const double y0 = _lats[j0], y1 = _lats[j1];
    const double dy = y1 - y0;
    const double t = (-90.0 - y0) / dy;

    double sum = 0.0;
    for (int i = 0; i < _nlon; ++i)
    {
      const double v0 = field[Idx(i, j0)];
      const double v1 = field[Idx(i, j1)];
      sum += v0 * (1.0 - t) + v1 * t;
    }
    return sum / _nlon;
  }

  double Bilerp(const std::vector<double> &field, double lon, double lat) const
  {
    if (_nlon <= 1 || _nlat <= 1)
      throw std::runtime_error("LonLatGrid::Bilerp requires nlon>1 and nlat>1");

    if (lat > 90.0)
      lat = 90.0;
    if (lat < -90.0)
      lat = -90.0;

    {
      double x = std::fmod(lon + 180.0, 360.0);
      if (x < 0.0)
        x += 360.0;
      lon = x - 180.0;
    }

    const double lonMin = _lons.front();
    const double lonMax = _lons.back();

    int i0, i1;
    double a;

    if (lon >= lonMin && lon <= lonMax)
    {
      int i_hi = int(std::lower_bound(_lons.begin(), _lons.end(), lon) - _lons.begin());
      if (i_hi == 0)
      {
        i0 = 0;
        i1 = 1;
      }
      else if (i_hi >= _nlon)
      {
        i0 = _nlon - 2;
        i1 = _nlon - 1;
      }
      else
      {
        i0 = i_hi - 1;
        i1 = i_hi;
      }
      const double x0 = _lons[i0], x1 = _lons[i1];
      a = (x1 != x0) ? (lon - x0) / (x1 - x0) : 0.0;
    }
    else
    {
      i0 = _nlon - 1;
      i1 = 0;
      const double seamWidth = (lonMin + 360.0) - lonMax;
      if (lon > lonMax)
        a = (lon - lonMax) / seamWidth;
      else
        a = ((lon + 360.0) - lonMax) / seamWidth;
    }

    if (lat > _lats.back())
    {
      const int jt = _nlat - 1;
      const double y0 = _lats[jt];
      const double vTop = (1.0 - a) * field[Idx(i0, jt)] + a * field[Idx(i1, jt)];
      const double den = (90.0 - y0);
      const double t = (lat - y0) / den;
      return (1.0 - t) * vTop + t * NorthPole(field);
    }

    if (lat < _lats.front())
    {
      const int jb = 0;
      const double y1 = _lats[jb];
      const double vBottom = (1.0 - a) * field[Idx(i0, jb)] + a * field[Idx(i1, jb)];
      const double den = (y1 - (-90.0));
      const double t = (lat - (-90.0)) / den;
      return (1.0 - t) * SouthPole(field) + t * vBottom;
    }

    int j_hi = int(std::lower_bound(_lats.begin(), _lats.end(), lat) - _lats.begin());
    int j0, j1;
    if (j_hi == 0)
    {
      j0 = 0;
      j1 = 1;
    }
    else if (j_hi >= _nlat)
    {
      j0 = _nlat - 2;
      j1 = _nlat - 1;
    }
    else
    {
      j0 = j_hi - 1;
      j1 = j_hi;
    }

    const double y0 = _lats[j0], y1 = _lats[j1];
    const double b = (y1 != y0) ? (lat - y0) / (y1 - y0) : 0.0;

    const double f00 = field[Idx(i0, j0)];
    const double f10 = field[Idx(i1, j0)];
    const double f01 = field[Idx(i0, j1)];
    const double f11 = field[Idx(i1, j1)];

    const double w00 = (1.0 - a) * (1.0 - b);
    const double w10 = a * (1.0 - b);
    const double w01 = (1.0 - a) * b;
    const double w11 = a * b;

    return f00 * w00 + f10 * w10 + f01 * w01 + f11 * w11;
  }

private:
  std::vector<double> _lons, _lats;
  int _nlon = 0, _nlat = 0;
};

class SolutionInterpolator
{
public:
   SolutionInterpolator(mfem::ParMesh &pmesh_,
                        int point_ordering_ = mfem::Ordering::byNODES,
                        double boundary_dist_tol_ = 10.0)
      : pmesh(pmesh_),
        comm(pmesh_.GetComm()),
        myid(mfem::Mpi::WorldRank()),
        dim(pmesh_.Dimension()),
        point_ordering(point_ordering_),
        boundary_dist_tol(boundary_dist_tol_),
        finder(comm)
   {
      finder.Setup(pmesh);
      finder.SetDistanceToleranceForPointsFoundOnBoundary(boundary_dist_tol);
   }

   void Write(const std::string &out_dat) const
   {
      if (myid != 0) return;
      if (cols.empty()) return;

      const int n = cols[0]->Size();
      for (const auto *v : cols)
      {
         if (!v || v->Size() != n)
            throw std::runtime_error("Column size mismatch.");
      }

      std::ofstream ofs(out_dat);
      ofs.setf(std::ios::scientific);
      ofs << std::setprecision(12);

      for (int i = 0; i < n; ++i)
      {
         for (int c = 0; c < (int)cols.size(); ++c)
         {
            ofs << (*cols[c])[i];
            if (c + 1 < (int)cols.size()) ofs << " ";
         }
         ofs << "\n";
      }
   }

protected:
   void InterpolateAtPoints(const mfem::GridFunction &x,
                            mfem::Vector &vxyz,
                            mfem::Vector &interp)
   {
      finder.FindPoints(vxyz, point_ordering);
      finder.Interpolate(x, interp);
      if (interp.UseDevice()) interp.HostReadWrite();
      vxyz.HostReadWrite();
   }

protected:
   mfem::ParMesh &pmesh;
   MPI_Comm comm;
   int myid;
   int dim;
   int point_ordering;
   double boundary_dist_tol;
   mfem::FindPointsGSLIB finder;

   std::vector<const mfem::Vector*> cols;
};

class SolutionRadialInterpolator : public SolutionInterpolator
{
public:
   using SolutionInterpolator::SolutionInterpolator;

   void Interpolate(const mfem::GridFunction &x,
                    int n,
                    double r0, double r1,
                    double lon_deg = 0.0,
                    double lat_deg = 0.0)
   {
      const double lon = lon_deg * M_PI / 180.0;
      const double lat = lat_deg * M_PI / 180.0;

      const double ux = std::cos(lat) * std::cos(lon);
      const double uy = std::cos(lat) * std::sin(lon);
      const double uz = std::sin(lat);

      rvec.SetSize(n);
      val.SetSize(n);
      vxyz.SetSize(n * dim);

      for (int i = 0; i < n; ++i)
      {
         const double r = r0 + (r1 - r0) * double(i) / double(n - 1);
         rvec[i] = r;

         if (dim > 0) vxyz[i]         = r * ux;
         if (dim > 1) vxyz[i + n]     = r * uy;
         if (dim > 2) vxyz[i + 2 * n] = r * uz;
      }

      InterpolateAtPoints(x, vxyz, val);
      cols = { &rvec, &val };
   }

private:
   mfem::Vector rvec;
   mfem::Vector val;
   mfem::Vector vxyz;
};

class SolutionSphericalInterpolator : public SolutionInterpolator
{
public:
   using SolutionInterpolator::SolutionInterpolator;

   static LonLatGrid DefaultGrid(int nlon = 360, int nlat = 181)
   {
      if (nlon < 2 || nlat < 2)
         throw std::runtime_error("DefaultGrid: nlon and nlat must be >= 2.");

      std::vector<double> lons(nlon), lats(nlat);

      for (int i = 0; i < nlon; ++i)
         lons[i] = -180.0 + 360.0 * double(i) / double(nlon);

      for (int j = 0; j < nlat; ++j)
         lats[j] = -90.0 + 180.0 * double(j) / double(nlat - 1);

      return LonLatGrid(lons, lats);
   }

   void Interpolate(const mfem::GridFunction &x,
                    double R,
                    const LonLatGrid &grid = DefaultGrid())
   {
      if (dim != 3)
         throw std::runtime_error("Sphere interpolation requires dim=3.");

      const int nlon = grid.NLon();
      const int nlat = grid.NLat();
      const int npts = nlon * nlat;

      lon_deg.SetSize(npts);
      lat_deg.SetSize(npts);
      val.SetSize(npts);
      vxyz.SetSize(npts * dim);

      for (int j = 0; j < nlat; ++j)
      {
         const double latd = grid.LatAt(j);
         const double lat = latd * M_PI / 180.0;
         const double clat = std::cos(lat);
         const double slat = std::sin(lat);

         for (int i = 0; i < nlon; ++i)
         {
            const double lond = grid.LonAt(i);
            const double lon = lond * M_PI / 180.0;

            const int k = (int)grid.Idx(i, j);

            lon_deg[k] = lond;
            lat_deg[k] = latd;

            vxyz[k]          = R * clat * std::cos(lon);
            vxyz[k + npts]   = R * clat * std::sin(lon);
            vxyz[k + 2*npts] = R * slat;
         }
      }

      InterpolateAtPoints(x, vxyz, val);
      cols = { &lon_deg, &lat_deg, &val };
   }

private:
   mfem::Vector lon_deg;
   mfem::Vector lat_deg;
   mfem::Vector val;
   mfem::Vector vxyz;
};

