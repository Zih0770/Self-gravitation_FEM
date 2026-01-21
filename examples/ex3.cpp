// Sample run:
// mpirun -np 4 bin/ex3 -m data/Phobos.msh -o 2 -mth 1 -deg 32
#include "common.hpp"
#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

constexpr real_t pi = std::numbers::pi_v<real_t>;

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const int myid = Mpi::WorldRank();
  const int nprocs = Mpi::WorldSize();

  const char* mesh_file = "data/Phobos.msh";
  int order = 3;
  int deg = 32;
  int method = 1;

  real_t rho_dim = 1860.0;
  real_t G_dim = 6.6743e-11;

  real_t L_scale = 11e3;
  real_t RHO_scale = rho_dim;
  real_t T_scale = 1.0 / std::sqrt(G_dim * RHO_scale);

  real_t start_time, end_time, assembly_time, solver_time;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Gmsh mesh file (.msh).");
  args.AddOption(&order, "-o", "--order", "H1 FE order.");
  args.AddOption(&deg, "-deg", "--deg", "spherical-harmonic cutoff.");
  args.AddOption(
      &method, "-mth", "--method",
      "Solution method: 0 = Neumann, 1 = DtN, 2 = multipole, 10 = Dirichlet.");
  args.AddOption(&rho_dim, "-density", "--density",
                 "Constant density of Phobos.");
  args.AddOption(&L_scale, "-L", "--L", "Length scale.");
  args.AddOption(&T_scale, "-T", "--T", "Time scale.");
  args.AddOption(&RHO_scale, "-RHO", "--RHO", "Density scale.");
  args.Parse();
  if (!args.Good()) {
    if (myid == 0) args.PrintUsage(cout);
    return 1;
  }
  if (myid == 0) args.PrintOptions(cout);

  Nondimensionalisation nd(L_scale, T_scale, RHO_scale);
  const real_t rho_nd = nd.ScaleDensity(rho_dim);
  real_t G = -4.0 * pi * G_dim * RHO_scale * T_scale * T_scale;
  ConstantCoefficient G_coeff(G);
  if (myid == 0) {
    cout << "RHS dimensionless number G = " << G << endl;
  }

  Mesh mesh(mesh_file, 1, 1, true);
  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  const int dim = pmesh.Dimension();

  Array<int> dom_marker(pmesh.attributes.Max());
  dom_marker = 0;
  dom_marker[0] = 1;

  auto bdr_marker = Array<int>(pmesh.bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[1] = 1;

  H1_FECollection fec(order, dim);
  L2_FECollection dfec(order - 1, dim);
  ParFiniteElementSpace fes(&pmesh, &fec);

  std::unique_ptr<ParFiniteElementSpace> dfes;
  if (method == 2) {
    dfes = std::make_unique<ParFiniteElementSpace>(&pmesh, &dfec);
  }

  HYPRE_BigInt size = fes.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  ConstantCoefficient eps(0.001);
  ParBilinearForm as(&fes);
  as.AddDomainIntegrator(new DiffusionIntegrator());
  as.AddDomainIntegrator(new MassIntegrator(eps));
  as.Assemble();

  auto rho_coeff_1 = ConstantCoefficient(rho_nd);
  auto rho_coeff = PWCoefficient();
  rho_coeff.UpdateCoefficient(1, rho_coeff_1);
  ProductCoefficient rhs_coeff(G_coeff, rho_coeff);

  auto x = ParGridFunction(&fes);

  ParLinearForm b(&fes);

  b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
  b.Assemble();

  if (method == 2) {
    if (myid == 0) {
      start_time = MPI_Wtime();
    }
    auto c = PoissonMultipoleOperator(MPI_COMM_WORLD, dfes.get(), &fes, deg,
                                      dom_marker);
    c.Assemble();
    if (myid == 0) {
      end_time = MPI_Wtime();
      assembly_time = (end_time - start_time);
    }
    auto rhof = GridFunction(dfes.get());
    rhof.ProjectCoefficient(rhs_coeff);
    c.AddMult(rhof, b, -1);
  }

  // Solve
  x = 0.0;
  Array<int> ess_tdof_list{};
  if (method == 10) fes.GetEssentialTrueDofs(bdr_marker, ess_tdof_list);

  HypreParMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  HypreParMatrix As;
  as.FormSystemMatrix(ess_tdof_list, As);
  auto P = HypreBoomerAMG(As);

  auto solver = CGSolver(MPI_COMM_WORLD);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(5000);
  solver.SetPrintLevel(1);

  if (method == 1) {
    if (myid == 0) {
      start_time = MPI_Wtime();
    }
    auto c = PoissonDtNOperator(MPI_COMM_WORLD, &fes, deg);
    c.Assemble();
    if (myid == 0) {
      end_time = MPI_Wtime();
      assembly_time = (end_time - start_time);
    }
    auto C = c.RAP();
    auto D = SumOperator(&A, 1.0, &C, 1.0, false, false);
    solver.SetOperator(D);
    solver.SetPreconditioner(P);

    if (myid == 0) {
      start_time = MPI_Wtime();
    }
    solver.Mult(B, X);
    if (myid == 0) {
      end_time = MPI_Wtime();
      solver_time = (end_time - start_time);
    }

    a.RecoverFEMSolution(X, b, x);
    nd.UnscaleGravityPotential(x);

    auto coeffs = Vector();
    c.HarmonicCoefficients(x, coeffs);

    if (myid == 0) {
      auto i = 0;
      cout << showpos;
      for (auto l = 0; l <= deg; l++) {
        cout << l << " " << 0 << " " << coeffs(i++) << endl;
        for (auto m = 1; m <= l; m++) {
          cout << l << " " << -m << " " << coeffs(i++) << endl;
          cout << l << " " << m << " " << coeffs(i++) << endl;
        }
      }
    }
  } else {
    if (myid == 0) {
      start_time = MPI_Wtime();
    }
    solver.SetOperator(A);
    solver.SetPreconditioner(P);
    if (method == 10) {
      solver.Mult(B, X);
    } else {
      auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
      orthoSolver.SetSolver(solver);
      orthoSolver.Mult(B, X);
    }
    if (myid == 0) {
      end_time = MPI_Wtime();
      solver_time = (end_time - start_time);
    }

    a.RecoverFEMSolution(X, b, x);
    nd.UnscaleGravityPotential(x);
  }

  if (myid == 0) {
    std::cout << "Assembly time: " << assembly_time << " s" << std::endl;
    std::cout << "Solver time: " << solver_time << " s" << std::endl;
  }

  // Post-processing
  auto submesh = ParSubMesh::CreateFromDomain(pmesh, dom_marker);
  auto subfes = ParFiniteElementSpace(&submesh, &fec);
  auto subx = ParGridFunction(&subfes);
  submesh.Transfer(x, subx);

  {
    ostringstream mesh_name, sol_name;
    mesh_name << "results/mesh_Pho." << setfill('0') << setw(6) << myid;
    sol_name << "results/sol_Pho." << setfill('0') << setw(6) << myid;

    ofstream mesh_ofs(mesh_name.str().c_str());
    mesh_ofs.precision(12);
    submesh.Print(mesh_ofs);

    ofstream sol_ofs(sol_name.str().c_str());
    sol_ofs.precision(12);
    subx.Save(sol_ofs);
  }

  {
    SolutionSphericalInterpolator si(pmesh, mfem::Ordering::byNODES, 10.0);

    auto x0 = MeshCentroid(&mesh, 3);
    auto [found, same, radius] = SphericalBoundaryRadius(&mesh, bdr_marker, x0);
    const real_t R = radius;

    // If commented out, use the default 1x1 lon-lat grid
    /*const int nlon = 360;
    const int nlat = 181;
    std::vector<double> lons(nlon), lats(nlat);
    for (int i = 0; i < nlon; ++i) lons[i] = -180.0 + 360.0 * i / nlon;
    for (int j = 0; j < nlat; ++j) lats[j] = -90.0 + 180.0 * j / (nlat - 1);
    LonLatGrid grid(lons, lats);*/

    si.Interpolate(x, R/*, grid*/);
    si.Write("results/Phobos_phi_on_sphere.dat");
  }
}
