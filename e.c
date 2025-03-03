#include <petsc.h>

static char help[] = "Solves a series of linear systems using KSPHPDDM.\n\n";

// int gcrodr(int argc, char **args)
int main(int argc, char **args)
{
  Vec x, b; /* computed solution and RHS */
  Mat A;    /* linear system matrix */
  KSP ksp;  /* linear solver context */

  PetscInt i, j, nmat = 2;
  PetscViewer viewer;
  char dir[PETSC_MAX_PATH_LEN], dir_output[1000], dir_x[1000], name[256];
  PetscBool flg, reset = PETSC_FALSE;

  PetscInt iter;
  PetscReal rnorm;
  char filename[PETSC_MAX_PATH_LEN]; 
  char filename_total_iter[PETSC_MAX_PATH_LEN];
  char filename_total_rnorm[PETSC_MAX_PATH_LEN];

  PetscViewer filex;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscStrncpy(dir, ".", sizeof(dir)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-load_dir", dir, sizeof(dir), NULL));

  PetscCall(PetscStrncpy(dir_output, ".", sizeof(dir_output)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-load_dir_output", dir_output, sizeof(dir_output), NULL));

  PetscCall(PetscStrncpy(dir_x, ".", sizeof(dir_x)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-load_dir_x", dir_x, sizeof(dir_x), NULL));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nmat", &nmat, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-reset", &reset, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  snprintf(filename_total_iter, PETSC_MAX_PATH_LEN, "%s/total/output_total_iter.txt", dir_output);
  FILE *file_total_iter = fopen(filename_total_iter, "w");
  snprintf(filename_total_rnorm, PETSC_MAX_PATH_LEN, "%s/total/output_total_rnorm.txt", dir_output);
  FILE *file_total_rnorm = fopen(filename_total_rnorm, "w");

  for (i = 0; i < nmat; i++)
  {
    j = i ;
    PetscCall(PetscSNPrintf(name, sizeof(name), "%s/A_%" PetscInt_FMT ".dat", dir, j));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
    PetscCall(MatLoad(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    if (i == 0)
      PetscCall(MatCreateVecs(A, &x, &b));
    PetscCall(PetscSNPrintf(name, sizeof(name), "%s/rhs_%" PetscInt_FMT ".dat", dir, j));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
    PetscCall(VecLoad(b, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(PetscObjectTypeCompare((PetscObject)ksp, KSPHPDDM, &flg));


    snprintf(filename, PETSC_MAX_PATH_LEN, "%s/output_%d.txt", dir_output, j);
    FILE *file = fopen(filename, "w");
    if (file != NULL)
    {
      KSPGetIterationNumber(ksp, &iter);
      KSPGetResidualNorm(ksp, &rnorm);
      fprintf(file, "Iterations: %d\n", iter);
      fprintf(file, "Residual norm: %g\n", rnorm);
      fclose(file);
      fprintf(file_total_iter, "%d\n", iter);
      fprintf(file_total_rnorm, "%g\n", rnorm);
    }
    else
    {
      PetscPrintf(PETSC_COMM_WORLD, "Error opening output file\n");
    }


    snprintf(filename, PETSC_MAX_PATH_LEN, "%s/x_%d.dat", dir_x, j);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &filex);
    VecView(x, filex);
    PetscViewerDestroy(&filex);

  }

  fclose(file_total_iter);
  fclose(file_total_rnorm);

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}