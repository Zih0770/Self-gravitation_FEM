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
